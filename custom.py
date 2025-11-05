import torch
import torch.nn as nn
from torchcrf import CRF
from spacy.tokens import Doc
from thinc.api import Model, PyTorchWrapper
from spacy.language import Language
from torch.nn.utils.rnn import pad_sequence

# import os
# os.environ = "1"

class BiLSTMCRF(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_labels):
    super(BiLSTMCRF, self).__init__()
    self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
    self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
    self.crf = CRF(num_labels, batch_first=True)

  def forward(self, embeddings, labels=None, mask=None):
    # Process embeddings through the BiLSTM
    lstm_out, _ = self.lstm(embeddings)  # lstm_out: (batch_size, sequence_length, hidden_dim * 2)
    emissions = self.hidden2tag(lstm_out)  # emissions: (batch_size, sequence_length, num_labels)

    # Ensure emissions have the correct shape
    if emissions.dim() != 3:
      raise ValueError(f"emissions must have dimension of 3, got {emissions.dim()}")

    # Compute loss during training
    if labels is not None:
      loss = -self.crf(emissions, labels, mask=mask)
      return loss

    # Decode predictions during inference
    return self.crf.decode(emissions, mask=mask)
    
@Language.factory("bilstm_crf_ner")
def create_bilstm_crf_ner(nlp, name, tok2vec_path, input_dim, hidden_dim, num_labels):
    # Load the pre-trained tok2vec component
    tok2vec = nlp.get_pipe("tok2vec")
    bilstm_crf = BiLSTMCRF(input_dim, hidden_dim, num_labels)
    model = PyTorchWrapper(bilstm_crf)
    return BiLSTMCRFNER(tok2vec, model)

class BiLSTMCRFNER:
    def __init__(self, tok2vec, model):
        self.tok2vec = tok2vec
        self.model = model

        # Define the label-to-index mapping
        self.label_to_index = {
            "O": 0,          # Outside of any entity
            "B-LOC": 1,      # Beginning of a location entity
            "I-LOC": 2,      # Inside a location entity
            "B-PER": 3,      # Beginning of a person entity
            "I-PER": 4,      # Inside a person entity
            "B-ORG": 5,      # Beginning of an organization entity
            "I-ORG": 6       # Inside an organization entity
        }

        # Optionally, create the reverse mapping for predictions
        self.index_to_label = {v: k for k, v in self.label_to_index.items()}

    def update(self, examples, drop=0.0, sgd=None, losses=None):
      if losses is None:
          losses = {}
      losses["ner"] = 0.0  # Initialize the NER loss

      # Detect the device (use GPU if available, otherwise CPU)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

      # Extract tok2vec embeddings for the batch of examples
      tok2vec_output = self.tok2vec.predict([example.predicted for example in examples])  # List of numpy arrays

      # Convert tok2vec_output to tensors and move to the correct device
      tok2vec_output = [torch.tensor(doc, dtype=torch.float32) for doc in tok2vec_output]

      # Pad the tok2vec_output to create a single tensor
      tok2vec_output = pad_sequence(tok2vec_output, batch_first=True)  # Shape: (batch_size, max_sequence_length, embedding_dim)

      # Generate labels for each example
      labels = [self._get_labels(example) for example in examples]  # List of label sequences

      # Determine the maximum sequence length in the batch
      max_length = max(len(label) for label in labels)

      # Pad the labels to the maximum sequence length
      padded_labels = []
      for label in labels:
          padded_labels.append(label + [-1] * (max_length - len(label)))  # Use -1 as the padding value

      # Replace -1 in padded_labels with 0 (or another valid label)
      padded_labels = torch.tensor(
          [[label if label != -1 else 0 for label in label_seq] for label_seq in padded_labels],
          dtype=torch.long
      )

      # Create a mask to indicate valid tokens (1 for valid, 0 for padding) using padded_labels
      mask = (padded_labels != 0)  # Shape: (batch_size, max_sequence_length)

      # Ensure the first timestep of the mask is valid for all sequences
      mask[:, 0] = True

      # Validate tensor shapes
      assert tok2vec_output.size(0) == padded_labels.size(0) == mask.size(0), "Batch sizes must match"
      assert tok2vec_output.size(1) == padded_labels.size(1) == mask.size(1), "Sequence lengths must match"
      assert tok2vec_output.size(2) > 0, "Embedding dimension must be greater than 0"

      # In custom.py, inside the update method, before line 106

      # --- DEBUGGING SNIPPET START ---
      # Move labels to CPU for inspection to avoid further CUDA errors
      cpu_labels = padded_labels.cpu()

      # Check if any label is out of the valid range [0, num_labels-1]
      # Also check for negative values in positions that are not masked out.
      # if torch.any(cpu_labels >= self.num_labels) or torch.any((cpu_labels < 0) & (mask.cpu() > 0)):
      #   print("---!!!--- VALIDATION ERROR DETECTED ---!!!---")
      #   print(f"Component '{self.name}' detected out-of-bounds labels.")
      #   print(f"Configured num_labels: {self.num_labels}")
      #   print(f"Label tensor shape: {cpu_labels.shape}")

      #   # Find the exact locations of the errors
      #   invalid_indices_upper = (cpu_labels >= self.num_labels).nonzero(as_tuple=False)
      #   invalid_indices_lower = ((cpu_labels < 0) & (mask.cpu() > 0)).nonzero(as_tuple=False)

      #   print(f"Max label found: {cpu_labels.max().item()}")
      #   print(f"Min label found (where mask is active): {cpu_labels[mask.cpu() > 0].min().item()}")

      #   if invalid_indices_upper.shape > 0:
      #     print(f"Found {invalid_indices_upper.shape} labels >= {self.num_labels}:")
      #     for idx in invalid_indices_upper[:5]: # Print first 5
      #       print(f"  - At index {tuple(idx.tolist())}, value: {cpu_labels[tuple(idx)].item()}")

      #   if invalid_indices_lower.shape > 0:
      #     print(f"Found {invalid_indices_lower.shape} negative labels in unmasked positions:")
      #     for idx in invalid_indices_lower[:5]: # Print first 5
      #       print(f"  - At index {tuple(idx.tolist())}, value: {cpu_labels[tuple(idx)].item()}")

      #   # Raise a clear exception to halt execution
      #   raise ValueError("Label index out of bounds. Halting training.")
# --- DEBUGGING SNIPPET END ---


      # Pass embeddings, labels, and mask as a tuple to the model, with is_train=True
      loss = self.model((tok2vec_output, padded_labels, mask), is_train=True)
      print(f"LOSS: {loss}")
      losses["ner"] += loss.item()  # Accumulate the loss
      loss.backward()

      if sgd:
          sgd.step()
      return losses

    def _convert_predictions_to_spans(self, doc, predictions):
        # Convert predictions to spaCy spans
        spans = []
        for start, end, label in predictions:
            spans.append(doc.char_span(start, end, label=label))
        return spans

    def _get_labels(self, example):
      # Convert gold-standard annotations to label indices
      labels = []
      for token in example.reference:
          if token.ent_iob_ == "O":
              labels.append(self.label_to_index["O"])  # Map "O" to its index
          else:
              label = f"{token.ent_iob_}-{token.ent_type_}"
              labels.append(self.label_to_index[label])  # Map label to index
      return labels