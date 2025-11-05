"""
Standalone Tagalog NER Pipeline Script using spaCy v3 and calamanCy

This script provides a complete workflow for training and evaluating Named Entity Recognition (NER)
models for Tagalog language using spaCy v3 and calamanCy.

CORRESPONDING project.yml COMMANDS:
------------------------------------
1. install_models()           -> install-models: Install calamanCy models
2. process_datasets()         -> process-datasets: Extract and prepare training data
3. train_baseline()           -> baseline: Train transition-based parser without embeddings
4. train_static_vectors()     -> static-vectors: Evaluate using calamanCy lg model
5. train_trf_monolingual()    -> trf-monolingual: Evaluate using calamanCy trf model
6. train_trf_multilingual()   -> trf-multilingual: Train transformer + BiLSTM-CRF with XLM-RoBERTa/mBERT
7. evaluate_model()           -> Used in multiple commands for evaluation
8. analyze_dataset()          -> analyze: Get dataset statistics

ADDITIONAL FUNCTIONS:
---------------------
- create_config(): Generate spaCy config files programmatically
- plot_results(): Visualization utilities for metrics
- convert_to_iob(): Convert spaCy format to IOB format
- get_model_info(): Display model metadata and performance

USAGE EXAMPLES:
---------------
# Install required models
python standalone_ner_pipeline.py --action install

# Process datasets
python standalone_ner_pipeline.py --action process --input assets/tlunified_ner.tar.gz

# Train baseline model
python standalone_ner_pipeline.py --action train-baseline --train corpus/train.spacy --dev corpus/dev.spacy --output training/baseline

# Train BiLSTM-CRF model (uses spaCy's built-in BILUO encoding)
python standalone_ner_pipeline.py --action train-bilstm-crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_crf

# Train BiLSTM-CRF with custom hyperparameters
python standalone_ner_pipeline.py --action train-bilstm-crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_crf --hidden-dim 512 --dropout 0.5

# Evaluate model
python standalone_ner_pipeline.py --action evaluate --model training/baseline/model-best --test corpus/test.spacy --output metrics/baseline-test.json

# Train with static vectors (calamanCy lg)
python standalone_ner_pipeline.py --action eval-static --test corpus/test.spacy --output metrics/static-vectors-test.json

# Train transformer + BiLSTM-CRF model
python standalone_ner_pipeline.py --action train-transformer --train corpus/train.spacy --dev corpus/dev.spacy --transformer xlm-roberta-base --output training/xlm-roberta

# Train transformer + BiLSTM-CRF with custom hyperparameters
python standalone_ner_pipeline.py --action train-transformer --train corpus/train.spacy --dev corpus/dev.spacy --transformer xlm-roberta-base --output training/xlm-roberta --hidden-dim 512 --dropout 0.5

# Analyze dataset
python standalone_ner_pipeline.py --action analyze --train corpus/train.spacy --dev corpus/dev.spacy

# Full benchmark workflow
python standalone_ner_pipeline.py --action benchmark

REQUIREMENTS:
-------------
- Python 3.8+
- spaCy >= 3.6.0
- calamanCy == 0.1.0
- spacy-transformers
- torch (for custom BiLSTM-CRF)
- torchcrf
- transformers[sentencepiece]

Install with: pip install -r requirements.txt
"""

import os
import sys
import json
import tarfile
import argparse
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import subprocess

import spacy
from spacy.cli.train import train as spacy_train
from spacy.cli.evaluate import evaluate as spacy_evaluate
from spacy.cli.assemble import assemble_cli
from spacy.tokens import DocBin, Doc
from spacy.training import Example
from spacy import util
from wasabi import msg
import typer

# Try importing optional dependencies
try:
    import torch
    import torch.nn as nn
    from torchcrf import CRF
    from torch.nn.utils.rnn import pad_sequence
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    msg.warn("PyTorch not available. BiLSTM-CRF training will be disabled.")

try:
    import calamancy
    CALAMANCY_AVAILABLE = True
except ImportError:
    CALAMANCY_AVAILABLE = False
    msg.warn("calamanCy not installed. Will attempt to install.")


# ============================================================================
# CUSTOM BiLSTM-CRF COMPONENTS
# ============================================================================

if TORCH_AVAILABLE:
    from spacy.language import Language
    from spacy.pipeline import TrainablePipe
    from thinc.api import Model, PyTorchWrapper, Config, chain, with_array, get_torch_default_device, set_dropout_rate
    from thinc.types import Floats2d, Ints1d, ArgsKwargs
    from thinc.shims.pytorch import PyTorchShim
    import numpy
    
    class BiLSTMCRF(nn.Module):
        """BiLSTM-CRF model for sequence labeling."""
        
        def __init__(self, input_dim: int, hidden_dim: int, num_labels: int, dropout: float = 0.3):
            super(BiLSTMCRF, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.num_labels = num_labels
            
            # BiLSTM layer
            self.lstm = nn.LSTM(
                input_dim, 
                hidden_dim, 
                num_layers=1,
                batch_first=True, 
                bidirectional=True,
                dropout=0.0  # No dropout for single layer
            )
            self.dropout = nn.Dropout(dropout)
            
            # Emission layer
            self.hidden2tag = nn.Linear(hidden_dim * 2, num_labels)
            
            # CRF layer
            self.crf = CRF(num_labels, batch_first=True)
        
        def forward(self, embeddings, labels=None, mask=None):
            """
            Forward pass through BiLSTM and CRF.
            
            Args:
                embeddings: (batch_size, seq_len, input_dim)
                labels: (batch_size, seq_len) - optional, for training
                mask: (batch_size, seq_len) - boolean mask for valid tokens
                
            Returns:
                If training (labels provided): negative log-likelihood loss
                If inference: decoded tag sequences
            """
            # BiLSTM forward pass
            lstm_out, _ = self.lstm(embeddings)
            lstm_out = self.dropout(lstm_out)
            
            # Compute emissions
            emissions = self.hidden2tag(lstm_out)  # (batch_size, seq_len, num_labels)
            
            # Validate dimensions
            if emissions.dim() != 3:
                raise ValueError(f"Emissions must have 3 dimensions, got {emissions.dim()}")
            
            # Training mode: compute CRF loss
            if labels is not None:
                # Clamp labels to valid range [0, num_labels-1]
                labels = torch.clamp(labels, 0, self.num_labels - 1)
                
                if mask is None:
                    mask = torch.ones(labels.shape, dtype=torch.bool, device=labels.device)
                loss = -self.crf(emissions, labels, mask=mask, reduction='mean')
                return loss
            
            # Inference mode: decode best path
            if mask is None:
                mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
            decoded = self.crf.decode(emissions, mask=mask)
            return decoded
    
    from spacy.training import offsets_to_biluo_tags, biluo_tags_to_offsets
    
    class BiLSTMCRFNER(TrainablePipe):
        """
        spaCy pipeline component for BiLSTM-CRF NER.
        Uses spaCy's built-in BILUO encoding utilities.
        """
        
        def __init__(self, vocab, model, name="bilstm_crf_ner", *, tok2vec=None, transformer=None, labels=None):
            # Call parent constructor first
            # TrainablePipe.__init__ is needed for proper spaCy integration
            # Note: We can't use super() here because TrainablePipe is a Cython class
            self.vocab = vocab
            self.model = model
            self.name = name
            self.cfg = {}
            self.tok2vec = tok2vec  # Store reference to tok2vec
            self.transformer = transformer  # Store reference to transformer
            
            # BILUO label mapping - let spaCy handle the scheme
            self._label_map = {}
            self._idx_to_label = {}
            
            # Debug logging control for repaired sequences
            self._repair_count = 0
            self._max_repair_logs = 5  # Log first 5 repairs for debugging
            
            if labels:
                self._initialize_labels(labels)
        
        def _initialize_labels(self, labels):
            """Initialize BILUO label mappings from entity types.
            Uses spaCy's standard BILUO scheme: B-, I-, L-, U-, O
            """
            biluo_labels = ["O"]  # Outside
            
            for label in sorted(labels):
                biluo_labels.extend([
                    f"B-{label}",  # Begin
                    f"I-{label}",  # Inside
                    f"L-{label}",  # Last
                    f"U-{label}",  # Unit (single token entity)
                ])
            
            self._label_map = {label: idx for idx, label in enumerate(biluo_labels)}
            self._idx_to_label = {idx: label for label, idx in self._label_map.items()}
            
            msg.info(f"Initialized {len(biluo_labels)} BILUO labels for {len(labels)} entity types")
            msg.info(f"  Entity types: {sorted(labels)}")
            msg.info(f"  Label map size: {len(self._label_map)}, idx->label size: {len(self._idx_to_label)}")
        
        @property
        def labels(self):
            """Return entity type labels (not BILUO tags)."""
            entity_labels = set()
            for label in self._label_map.keys():
                if label != "O" and "-" in label:
                    entity_labels.add(label.split("-", 1)[1])
            return tuple(sorted(entity_labels))
        
        def _spans_to_biluo(self, doc):
            """Convert spaCy spans to BILUO tags using spaCy's built-in function."""
            # Get entity offsets
            entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
            
            # Use spaCy's built-in converter
            biluo_tags = offsets_to_biluo_tags(doc, entities)
            
            # Convert to indices
            return [self._label_map.get(tag, 0) for tag in biluo_tags]
        
        def _biluo_to_spans(self, doc, tag_indices):
            """Convert BILUO tag indices to spaCy spans using spaCy's built-in function."""
            # Convert indices back to tag strings
            biluo_tags = [self._idx_to_label.get(idx, "O") for idx in tag_indices]

            # Repair invalid BILUO sequences (sometimes model predictions can be invalid)
            def _repair_biluo(tags: List[str]) -> List[str]:
                repaired = list(tags)
                for i, tag in enumerate(repaired):
                    if tag == "O":
                        continue
                    if "-" not in tag:
                        # unexpected format, treat as O
                        repaired[i] = "O"
                        continue
                    prefix, label = tag.split("-", 1)
                    if prefix == "U":
                        continue
                    if prefix == "B":
                        continue
                    if prefix == "I":
                        # I- should follow B- or I- of same label
                        if i == 0 or not repaired[i-1].endswith(f"-{label}") or not repaired[i-1].startswith(("B-", "I-")):
                            repaired[i] = f"B-{label}"
                    if prefix == "L":
                        # L- should follow B- or I- of same label; otherwise convert to U-
                        if i == 0 or not repaired[i-1].endswith(f"-{label}") or not repaired[i-1].startswith(("B-", "I-")):
                            repaired[i] = f"U-{label}"
                return repaired

            repaired_tags = _repair_biluo(biluo_tags)

            # Log first N repairs for debugging
            if repaired_tags != biluo_tags and self._repair_count < self._max_repair_logs:
                self._repair_count += 1
                msg.warn(f"[Repair #{self._repair_count}] Invalid BILUO sequence detected and repaired:")
                msg.text(f"  Original: {biluo_tags[:20]}{'...' if len(biluo_tags) > 20 else ''}")
                msg.text(f"  Repaired: {repaired_tags[:20]}{'...' if len(repaired_tags) > 20 else ''}")
                msg.text(f"  Document text: {doc.text[:100]}...")

            # Use spaCy's built-in converter on repaired tags
            entities = biluo_tags_to_offsets(doc, repaired_tags)
            
            # Convert to (start_token, end_token, label) format
            spans = []
            for start_char, end_char, label in entities:
                span = doc.char_span(start_char, end_char, label=label)
                if span is not None:
                    spans.append((span.start, span.end, label))
            
            return spans
        
        def predict(self, docs):
            """Predict entities for a batch of documents."""
            if not docs:
                return []
            
            # Get PyTorch model and set to eval mode
            pytorch_model = self.model.attrs["pytorch_model"]
            pytorch_model.eval()
            device = next(pytorch_model.parameters()).device
            
            # Validate label map consistency
            if pytorch_model.num_labels != len(self._label_map):
                msg.fail(f"CRITICAL: Model has {pytorch_model.num_labels} labels but component has {len(self._label_map)} labels in label_map!")
                raise ValueError(f"Label count mismatch: model={pytorch_model.num_labels}, component={len(self._label_map)}")
            
            # Get embeddings from tok2vec (which may be a TransformerListener)
            if self.tok2vec is not None:
                # Call tok2vec.predict() - if it's a TransformerListener, it will get data from doc._.trf_data
                # which was set by the upstream Transformer component
                tokvecs_output = self.tok2vec.predict(docs)
                
                # tokvecs_output is List[Floats2d] from the TransformerListener + pooling chain
                # Convert to numpy arrays
                tokvecs = []
                for tokvec_array in tokvecs_output:
                    # Convert CuPy to NumPy if needed
                    if hasattr(tokvec_array, 'get'):  # CuPy array
                        tokvec_np = tokvec_array.get()
                    else:
                        tokvec_np = numpy.asarray(tokvec_array)
                    
                    tokvecs.append(tokvec_np)
            elif self.transformer is not None:
                # Fallback: Use external transformer embeddings (feature-based, for backward compatibility)
                transformer_batch = self.transformer.predict(docs)
                
                # Extract tensors from FullTransformerBatch
                tokvecs = []
                for doc_data in transformer_batch.doc_data:
                    # Get token embeddings and convert to numpy
                    tensor = doc_data.tensors[0]  # Shape: (seq_len, hidden_dim) or (1, seq_len, hidden_dim)
                    
                    # Convert CuPy to NumPy if needed
                    if hasattr(tensor, 'get'):  # CuPy array
                        tensor = tensor.get()
                    
                    # Ensure 2D: (seq_len, hidden_dim)
                    if hasattr(tensor, 'ndim') and tensor.ndim == 3:
                        # Only squeeze if first dimension is actually 1
                        if tensor.shape[0] == 1:
                            tensor = numpy.squeeze(tensor, axis=0)
                        else:
                            # If first dimension is not 1, take first element (shouldn't happen but be safe)
                            tensor = tensor[0]
                    
                    tokvecs.append(tensor)
            else:
                raise ValueError("Either tok2vec or transformer must be available")
            
            # Process each document
            predictions = []
            with torch.no_grad():
                for doc, tokvec in zip(docs, tokvecs):
                    # Prepare embeddings
                    seq_len = min(len(doc), len(tokvec))
                    if seq_len == 0:
                        predictions.append([])
                        continue
                    
                    tokvec = tokvec[:seq_len]
                    embeddings = torch.tensor(tokvec, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    # Get predictions from model (returns list of tag sequences)
                    decoded = pytorch_model(embeddings)  # No labels = inference mode
                    
                    # decoded is a list of lists, get first (and only) sequence
                    if decoded and len(decoded) > 0:
                        predictions.append(decoded[0])
                    else:
                        predictions.append([0] * seq_len)  # Default to all 'O' tags
            
            return predictions
        
        def set_annotations(self, docs, predictions):
            """Set entity annotations on documents."""
            for doc, tags in zip(docs, predictions):
                # Convert BILUO tags to spans
                spans = self._biluo_to_spans(doc, tags)
                
                # Create entity spans
                ents = []
                for start, end, label in spans:
                    span = doc[start:end]
                    span.label_ = label
                    ents.append(span)
                
                # Set entities (filter out overlaps)
                try:
                    doc.ents = ents
                except ValueError:
                    # Handle overlapping entities
                    doc.ents = util.filter_spans(ents)
        
        def score(self, examples, **kwargs):
            """Score a batch of examples and return the scores.
            
            This method is called during evaluation to compute metrics like precision, recall, and F1.
            Without this method, spaCy's training loop shows zero scores.
            """
            from spacy.scorer import Scorer
            
            # Use spaCy's built-in NER scorer
            scorer = Scorer()
            return scorer.score_spans(examples, "ents", **kwargs)
        
        def update(self, examples, *, drop=0.0, sgd=None, losses=None):
            """Update the model on a batch of examples with gradient backpropagation through tok2vec."""
            if losses is None:
                losses = {}
            losses.setdefault(self.name, 0.0)
            
            if not examples:
                return losses
            
            # Get PyTorch model
            pytorch_model = self.model.attrs["pytorch_model"]
            device = next(pytorch_model.parameters()).device
            
            # Validate label map consistency
            if pytorch_model.num_labels != len(self._label_map):
                msg.fail(f"CRITICAL: Model has {pytorch_model.num_labels} labels but component has {len(self._label_map)} labels in label_map!")
                raise ValueError(f"Label count mismatch during update: model={pytorch_model.num_labels}, component={len(self._label_map)}")
            
            # Set dropout
            pytorch_model.train()
            
            # Create optimizer if needed
            if not hasattr(self, '_pytorch_optimizer'):
                self._pytorch_optimizer = torch.optim.Adam(pytorch_model.parameters(), lr=0.001)
            
            # Prepare training data
            docs = [eg.predicted for eg in examples]
            gold_docs = [eg.reference for eg in examples]
            
            # Get embeddings and gradient callback from tok2vec (which may be a TransformerListener)
            if self.tok2vec is not None:
                # Call begin_update on tok2vec model
                # self.tok2vec might be either:
                # 1. A component (from get_pipe) - use .model
                # 2. A Model directly (from TransformerListener pattern) - use directly
                tok2vec_model = self.tok2vec.model if hasattr(self.tok2vec, 'model') else self.tok2vec
                tokvecs, bp_tokvecs = tok2vec_model.begin_update(docs)
                
                # tokvecs is Floats2d or List[Floats2d], convert to list of torch tensors
                tokvecs_torch_list = []
                
                # Check if it's a single array or list of arrays
                if hasattr(tokvecs, '__iter__') and not hasattr(tokvecs, 'shape'):
                    # It's a list of arrays
                    tokvec_list = tokvecs
                else:
                    # It's a single array, need to split by docs
                    # This shouldn't happen with TransformerListener, but handle it
                    tokvec_list = [tokvecs]
                
                for tokvec_array in tokvec_list:
                    # Convert to torch tensor
                    if hasattr(tokvec_array, 'get'):
                        # CuPy array
                        tensor_np = tokvec_array.get()
                        torch_tensor = torch.from_numpy(tensor_np).to(device)
                    elif isinstance(tokvec_array, torch.Tensor):
                        torch_tensor = tokvec_array.to(device)
                    else:
                        # NumPy array
                        torch_tensor = torch.from_numpy(numpy.asarray(tokvec_array)).to(device)
                    
                    torch_tensor.requires_grad_(True)
                    tokvecs_torch_list.append(torch_tensor)
                
            elif self.transformer is not None:
                # IMPORTANT: For fine-tuning transformers in spaCy, you MUST use the TransformerListener
                # architecture. Direct calls to begin_update() are not supported for custom components.
                # As a workaround for now, we'll use feature-based approach (no fine-tuning).
                # Read embeddings from doc._.trf_data which the transformer component already populated.
                tokvecs_torch_list = []
                for doc in docs:
                    if not hasattr(doc._, 'trf_data') or doc._.trf_data is None:
                        raise ValueError(f"Transformer component must run before {self.name}")
                    # Get the transformer embeddings (already computed by transformer component)
                    tensor = doc._.trf_data.tensors[0]  # [seq_len, hidden_dim]

                    # Get the transformer embeddings (already computed by transformer component)
                    tensor = doc._.trf_data.tensors[0]  # [seq_len, hidden_dim]
                    
                    # Convert to torch tensor
                    if hasattr(tensor, 'get'):
                        # CuPy array
                        tensor_np = tensor.get()
                        torch_tensor = torch.from_numpy(tensor_np).to(device)
                    elif isinstance(tensor, torch.Tensor):
                        torch_tensor = tensor.to(device)
                    else:
                        torch_tensor = torch.from_numpy(tensor).to(device)
                    
                    tokvecs_torch_list.append(torch_tensor)
                
                #NO backprop to transformer in this simplified version
                bp_tokvecs = None
            elif self.tok2vec is not None:
                # Update tok2vec first so it reports its loss
                if sgd is not None:
                    self.tok2vec.update(examples, drop=drop, sgd=sgd, losses=losses)
                
                # Get tok2vec embeddings in TRAINING mode with gradient tracking
                tokvecs, bp_tokvecs = self.tok2vec.model.begin_update(docs)
                
                # Convert to numpy/cupy for processing
                if hasattr(tokvecs, 'get'):
                    tokvecs_np = tokvecs.get()
                else:
                    tokvecs_np = tokvecs
            else:
                raise ValueError("Either tok2vec or transformer must be available")
            
            # Collect gradients for embedding layer
            tokvec_gradients = []
            
            # Zero PyTorch gradients
            self._pytorch_optimizer.zero_grad()
            
            # Accumulate loss and gradients across the batch
            total_loss = 0.0
            num_valid_docs = 0
            
            # Use transformer torch tensors for fine-tuning
            if hasattr(self, '_tok2vec_transformer') and self._tok2vec_transformer is not None:
                iter_source = zip(docs, tokvecs_torch_list, gold_docs)
            elif self.transformer is not None:
                iter_source = zip(docs, tokvecs_torch_list, gold_docs)
            else:
                # tok2vec uses numpy/cupy
                iter_source = zip(docs, tokvecs_np, gold_docs)

            for doc_idx, (doc, tokvec, gold_doc) in enumerate(iter_source):
                # Get gold BILUO labels
                gold_biluo = self._spans_to_biluo(gold_doc)
                
                # Truncate to actual doc length
                if hasattr(self, '_tok2vec_transformer') and self._tok2vec_transformer is not None:
                    seq_len = min(len(doc), tokvec.size(0), len(gold_biluo))
                elif self.transformer is not None:
                    seq_len = min(len(doc), tokvec.size(0), len(gold_biluo))
                else:
                    seq_len = min(len(doc), len(tokvec), len(gold_biluo))
                
                if seq_len == 0:
                    # Skip empty sequences
                    if self.transformer is None and not hasattr(self, '_tok2vec_transformer'):
                        tokvec_gradients.append(numpy.zeros_like(tokvec))
                    continue
                
                gold_biluo = gold_biluo[:seq_len]
                
                # Prepare embeddings based on component type
                if hasattr(self, '_tok2vec_transformer') and self._tok2vec_transformer is not None:
                    # tokvec is already a torch.Tensor from tok2vec transformer
                    t_seq = tokvec[:seq_len].to(device)
                    # Add batch dimension
                    embeddings = t_seq.unsqueeze(0)
                    # Retain gradients so we can read them after backward
                    embeddings.retain_grad()
                elif self.transformer is not None:
                    # tokvec is already a torch.Tensor from transformer
                    t_seq = tokvec[:seq_len].to(device)
                    # Add batch dimension
                    embeddings = t_seq.unsqueeze(0)
                    # Retain gradients so we can read them after backward
                    embeddings.retain_grad()
                else:
                    # tok2vec: convert numpy/cupy to torch tensor
                    if hasattr(tokvec, 'get'):
                        tokvec = tokvec.get()
                    
                    tokvec = tokvec[:seq_len]
                    if isinstance(tokvec, numpy.ndarray):
                        if tokvec.ndim == 3:
                            tokvec = numpy.squeeze(tokvec, axis=0)
                        embeddings = torch.tensor(tokvec, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)
                    else:
                        embeddings = torch.tensor(tokvec, dtype=torch.float32, device=device, requires_grad=True).unsqueeze(0)

                    embeddings.retain_grad()
                
                labels = torch.tensor(gold_biluo, dtype=torch.long, device=device).unsqueeze(0)
                
                # Clamp labels to valid range
                num_labels = len(self._label_map)
                labels = torch.clamp(labels, 0, num_labels - 1)
                
                # Forward pass
                loss = pytorch_model(embeddings, labels)
                
                # Backward pass to get gradients
                loss.backward()
                
                # Extract and handle gradients based on component type
                if embeddings.grad is not None:
                    if hasattr(self, '_tok2vec_transformer') and self._tok2vec_transformer is not None:
                        # For embedded transformer: prepare gradient and accumulate
                        grad_torch = embeddings.grad.squeeze(0)  # (seq_len, input_dim)
                        tokvec_gradients.append(grad_torch)
                    elif self.transformer is not None:
                        # For transformer: prepare gradient as torch tensor
                        # and accumulate it to backprop through transformer later
                        grad_torch = embeddings.grad.squeeze(0)  # (seq_len, input_dim)
                        
                        # Store the gradient (keeping it as a torch tensor)
                        tokvec_gradients.append(grad_torch)
                    else:
                        # For tok2vec: convert to numpy/cupy as before
                        grad = embeddings.grad.squeeze(0).cpu().numpy()
                        
                        if hasattr(tokvecs_np[doc_idx], 'get'):
                            import cupy
                            full_grad = cupy.zeros_like(tokvecs_np[doc_idx])
                            full_grad[:seq_len] = cupy.array(grad)
                        else:
                            full_grad = numpy.zeros_like(tokvecs_np[doc_idx])
                            full_grad[:seq_len] = grad
                        
                        tokvec_gradients.append(full_grad)
                else:
                    # Zero gradient
                    if hasattr(self, '_tok2vec_transformer') and self._tok2vec_transformer is not None:
                        # Zero torch tensor gradient for embedded transformer
                        tokvec_gradients.append(torch.zeros_like(tokvecs_torch_list[doc_idx]))
                    elif self.transformer is not None:
                        # Zero torch tensor gradient
                        tokvec_gradients.append(torch.zeros_like(tokvecs_torch_list[doc_idx]))
                    elif hasattr(tokvecs_np[doc_idx], 'get'):
                        import cupy
                        tokvec_gradients.append(cupy.zeros_like(tokvecs_np[doc_idx]))
                    else:
                        tokvec_gradients.append(numpy.zeros_like(tokvecs_np[doc_idx]))
                
                total_loss += loss.item()
                num_valid_docs += 1
            
            # Update PyTorch parameters (BiLSTM-CRF)
            if num_valid_docs > 0:
                self._pytorch_optimizer.step()
                losses[self.name] += total_loss / num_valid_docs
            
            # Backpropagate gradients to tok2vec (which will pass to transformer if using TransformerListener)
            if tokvec_gradients and num_valid_docs > 0 and bp_tokvecs is not None:
                # Convert torch gradients back to numpy/cupy for Thinc
                # Important: Use same array type (cupy/numpy) as the input to maintain consistency
                grad_arrays = []
                for grad_item in tokvec_gradients:
                    if isinstance(grad_item, torch.Tensor):
                        # Torch tensor - convert to cupy if on GPU, numpy otherwise
                        if grad_item.is_cuda:
                            # Convert to CuPy array (stay on GPU)
                            import cupy
                            grad_cupy = cupy.asarray(grad_item.detach())
                            grad_arrays.append(grad_cupy)
                        else:
                            # Convert to numpy (CPU)
                            grad_np = grad_item.detach().cpu().numpy()
                            grad_arrays.append(grad_np)
                    else:
                        # Already numpy/cupy
                        grad_arrays.append(grad_item)
                
                # Call the backprop callback - this will:
                # 1. Pass gradients back through the pooling layer
                # 2. If tok2vec is a TransformerListener, accumulate gradients to send to upstream Transformer
                # 3. The upstream Transformer will collect gradients from all listeners and update
                bp_tokvecs(grad_arrays)
                
                # Finish update on tok2vec model (calls optimizer if provided)
                if sgd is not None:
                    tok2vec_model.finish_update(sgd)
            
            return losses
        
        def initialize(self, get_examples, *, nlp=None, labels=None):
            """Initialize the model and labels.
            
            This method must be defined to override TrainablePipe.initialize.
            """
            if labels is not None:
                self._initialize_labels(labels)
            else:
                # Extract labels from examples
                entity_labels = set()
                for example in get_examples():
                    for ent in example.reference.ents:
                        entity_labels.add(ent.label_)
                self._initialize_labels(sorted(entity_labels))
            
            # Get tok2vec or transformer from nlp if available
            if nlp is not None:
                if "tok2vec" in nlp.pipe_names:
                    self.tok2vec = nlp.get_pipe("tok2vec")
                if "transformer" in nlp.pipe_names:
                    self.transformer = nlp.get_pipe("transformer")
            
            # Recreate PyTorch model with correct number of labels
            # This is necessary because the model might have been created with wrong num_labels
            num_labels = len(self._label_map)
            if num_labels > 0:
                old_pytorch_model = self.model.attrs.get("pytorch_model")
                if old_pytorch_model:
                    # Get model parameters
                    input_dim = old_pytorch_model.input_dim
                    hidden_dim = old_pytorch_model.hidden_dim
                    # Dropout is stored as nn.Dropout module, extract p parameter
                    dropout = old_pytorch_model.dropout.p if hasattr(old_pytorch_model, 'dropout') else 0.3
                    
                    old_num_labels = old_pytorch_model.num_labels
                    msg.info(f"Recreating model: old num_labels={old_num_labels}, new num_labels={num_labels}")
                    
                    # Validate consistency
                    if old_num_labels != num_labels:
                        msg.warn(f"Label count mismatch! Factory created model with {old_num_labels} labels, "
                                f"but initialize() found {num_labels} labels. Recreating model.")
                    
                    # Create new model with correct num_labels
                    new_pytorch_model = BiLSTMCRF(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        num_labels=num_labels,
                        dropout=dropout
                    )
                    
                    # Update the model reference
                    self.model.attrs["pytorch_model"] = new_pytorch_model
                    
                    # Move to appropriate device
                    device = next(old_pytorch_model.parameters()).device
                    new_pytorch_model.to(device)
                    
                    msg.good(f"Model recreated successfully with {num_labels} labels")
            
            # Initialize tok2vec dimensions
            # For TransformerListener, we need to manually set the output dimension
            # because the listener is lazy and won't set it until first forward pass
            if self.tok2vec is not None:
                try:
                    # The tok2vec is a chain: TransformerListener -> Pooling
                    # We need to set the dimension on the inner TransformerListener
                    if hasattr(self.tok2vec, 'layers') and len(self.tok2vec.layers) > 0:
                        # It's a chain model, set dimension on first layer (the listener)
                        listener = self.tok2vec.layers[0]
                        if hasattr(listener, 'set_dim'):
                            # Set the listener's output dimension to match transformer output
                            # XLM-RoBERTa base outputs 768 dimensions
                            listener.set_dim("nO", 768, force=True)
                            msg.good("Manually set TransformerListener output dimension to 768")
                        
                        # Also set the final output dimension on the chain
                        if hasattr(self.tok2vec, 'set_dim'):
                            self.tok2vec.set_dim("nO", 768, force=True)
                    elif hasattr(self.tok2vec, 'set_dim'):
                        # Not a chain, set directly
                        self.tok2vec.set_dim("nO", 768, force=True)
                        msg.good("Manually set tok2vec output dimension to 768")
                    else:
                        # For regular tok2vec, try to get dimension
                        try:
                            dim = self.tok2vec.get_dim("nO")
                            msg.info(f"Tok2vec output dimension: {dim}")
                        except Exception as e:
                            msg.warn(f"Could not get tok2vec dimension: {e}")
                except Exception as e:
                    msg.warn(f"Could not initialize tok2vec dimensions: {e}")
        
        def to_disk(self, path, *, exclude=tuple()):
            """Serialize the component to disk."""
            path = Path(path)
            if not path.exists():
                path.mkdir(parents=True)
            
            # Save label mappings
            with (path / "labels.json").open("w") as f:
                json.dump({
                    "label_map": self._label_map,
                    "idx_to_label": self._idx_to_label
                }, f, indent=2)
            
            # Save model
            pytorch_model = self.model.attrs["pytorch_model"]
            torch.save(pytorch_model.state_dict(), path / "model.pt")
            
            # Save optimizer state if it exists
            if hasattr(self, '_pytorch_optimizer'):
                torch.save(self._pytorch_optimizer.state_dict(), path / "optimizer.pt")
        
        def from_disk(self, path, *, exclude=tuple()):
            """Load the component from disk."""
            path = Path(path)
            
            # Load label mappings FIRST
            with (path / "labels.json").open("r") as f:
                label_data = json.load(f)
                self._label_map = label_data["label_map"]
                self._idx_to_label = {int(k): v for k, v in label_data["idx_to_label"].items()}
            
            msg.info(f"Loaded {len(self._label_map)} labels from disk")
            
            # Recreate model with correct num_labels BEFORE loading weights
            num_labels = len(self._label_map)
            old_pytorch_model = self.model.attrs["pytorch_model"]
            
            # Get model parameters
            input_dim = old_pytorch_model.input_dim
            hidden_dim = old_pytorch_model.hidden_dim
            dropout = old_pytorch_model.dropout.p if hasattr(old_pytorch_model, 'dropout') else 0.3
            device = next(old_pytorch_model.parameters()).device
            
            msg.info(f"Recreating model from disk: num_labels={num_labels}, input_dim={input_dim}, hidden_dim={hidden_dim}")
            
            # Create new model with correct num_labels
            pytorch_model = BiLSTMCRF(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_labels=num_labels,
                dropout=dropout
            )
            pytorch_model.to(device)
            
            # Update model reference
            self.model.attrs["pytorch_model"] = pytorch_model
            
            # NOW load the weights
            pytorch_model.load_state_dict(torch.load(path / "model.pt"))
            
            # Load optimizer state if it exists
            optimizer_path = path / "optimizer.pt"
            if optimizer_path.exists() and hasattr(self, '_pytorch_optimizer'):
                self._pytorch_optimizer.load_state_dict(torch.load(optimizer_path))
            
            return self


    def build_bilstm_crf_model(input_dim, hidden_dim=256, dropout=0.3, num_labels=13):
        """Build a simple PyTorchWrapper model for BiLSTM-CRF."""
        # Create PyTorch model
        pytorch_model = BiLSTMCRF(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout
        )
        
        # Wrap with PyTorchWrapper - this handles Thinc integration automatically
        model = PyTorchWrapper(pytorch_model)
        model.attrs["pytorch_model"] = pytorch_model
        
        return model
    
    @Language.factory(
        "bilstm_crf_ner",
        default_config={
            "hidden_dim": 256,
            "dropout": 0.3,
            "labels": None
        }
    )
    def make_bilstm_crf_ner(nlp, name, hidden_dim, dropout, labels):
        """Factory function to create BiLSTM-CRF NER component."""
        # Get tok2vec or transformer from pipeline
        tok2vec = None
        transformer = None
        input_dim = 256  # Default
        
        if "tok2vec" in nlp.pipe_names:
            tok2vec = nlp.get_pipe("tok2vec")
            # Get input dimension from tok2vec
            try:
                input_dim = tok2vec.model.get_dim("nO")
            except:
                input_dim = 256
        elif "transformer" in nlp.pipe_names:
            transformer = nlp.get_pipe("transformer")
            # Get input dimension from transformer
            try:
                input_dim = transformer.model.get_dim("nO")
            except:
                # Default transformer output dimension (e.g., 768 for BERT-base, XLM-R-base)
                input_dim = 768
        
        # Determine number of labels
        if labels:
            entity_labels = labels
        else:
            entity_labels = []
        
        # Calculate BILUO label count: O + 4 tags per entity type
        num_labels = 1 + (len(entity_labels) * 4) if entity_labels else 1
        
        # Build Thinc model with proper integration
        model = build_bilstm_crf_model(input_dim, hidden_dim, dropout, num_labels)
        
        # Create component
        return BiLSTMCRFNER(nlp.vocab, model, name=name, tok2vec=tok2vec, transformer=transformer, labels=entity_labels)


    @Language.factory(
        "bilstm_crf_ner_trf",
        default_config={
            "hidden_dim": 256,
            "dropout": 0.3,
            "labels": None,
            "tok2vec": {
                "@architectures": "spacy-transformers.TransformerListener.v1",
                "grad_factor": 1.0,
                "pooling": {
                    "@layers": "reduce_mean.v1"
                },
                "upstream": "*"
            }
        }
    )
    def make_bilstm_crf_ner_trf(nlp, name, hidden_dim, dropout, labels, tok2vec):
        """
        Factory function to create BiLSTM-CRF NER component with TransformerListener.
        This enables true fine-tuning by connecting to a shared upstream Transformer component.
        
        The TransformerListener pattern allows:
        - Multiple components to share a single transformer
        - Gradients to flow back from this component to fine-tune the transformer
        - More efficient processing (transformer runs once per batch)
        
        Args:
            tok2vec: A Thinc Model that's already been resolved from the config.
                     This will be a TransformerListener model that connects to the upstream transformer.
        """
        # tok2vec is a TransformerListener + pooling chain at this point
        # (spaCy's config system resolves it before passing to factory)
        
        # Determine number of labels
        if labels:
            entity_labels = labels
        else:
            entity_labels = []
        
        # Calculate BILUO label count: O + 4 tags per entity type
        num_labels = 1 + (len(entity_labels) * 4) if entity_labels else 1
        
        # Get the transformer output dimension from the tok2vec listener
        input_dim = 768  # Standard for most base transformers
        try:
            # Try to get actual dimension from the model
            if hasattr(tok2vec, 'get_dim'):
                input_dim = tok2vec.get_dim("nO")
        except:
            pass  # Use default
        
        # Create the PyTorch BiLSTM-CRF model
        pytorch_bilstm_crf = BiLSTMCRF(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout
        )
        
        # Wrap in PyTorchWrapper
        pytorch_wrapper = PyTorchWrapper(pytorch_bilstm_crf)
        pytorch_wrapper.attrs["pytorch_model"] = pytorch_bilstm_crf
        
        # Create component that uses the TransformerListener tok2vec
        component = BiLSTMCRFNER(
            nlp.vocab,
            pytorch_wrapper,
            name=name,
            tok2vec=tok2vec,  # This is the TransformerListener that will connect to upstream transformer
            transformer=None,  # Not used with listener pattern
            labels=entity_labels
        )
        
        msg.info(f"Created BiLSTM-CRF NER component with TransformerListener tok2vec (input_dim={input_dim}, hidden_dim={hidden_dim}, num_labels={num_labels})")
        
        return component


# Import for component initialization
try:
    from itertools import islice
    import numpy as np
except ImportError:
    pass


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

def create_default_config(lang: str = "tl", use_gpu: bool = False) -> str:
    """
    Create a default spaCy configuration for NER training.
    
    Args:
        lang: Language code (default: "tl" for Tagalog)
        use_gpu: Whether to use GPU for training
        
    Returns:
        Configuration string in spaCy's config format
    """
    gpu_id = 0 if use_gpu else -1
    
    config = f"""
[paths]
train = null
dev = null
vectors = null
init_tok2vec = null

[system]
gpu_allocator = {"pytorch" if use_gpu else "null"}
seed = 0

[nlp]
lang = "{lang}"
pipeline = ["tok2vec","ner"]
batch_size = 1000
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {{"@tokenizers":"spacy.Tokenizer.v1"}}

[components]

[components.ner]
factory = "ner"
incorrect_spans_key = null
moves = null
scorer = {{"@scorers":"spacy.ner_scorer.v1"}}
update_with_oracle_cut_size = 100

[components.ner.model]
@architectures = "spacy.TransitionBasedParser.v2"
state_type = "ner"
extra_state_tokens = false
hidden_width = 64
maxout_pieces = 2
use_upper = true
nO = null

[components.ner.model.tok2vec]
@architectures = "spacy.Tok2VecListener.v1"
width = 256
upstream = "*"

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 256
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,1000,2500,2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 256
depth = 8
window_size = 1
maxout_pieces = 3

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.001

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vectors = ${{paths.vectors}}
init_tok2vec = ${{paths.init_tok2vec}}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""
    return config


def create_bilstm_crf_config(lang: str = "tl", use_gpu: bool = False, hidden_dim: int = 256, dropout: float = 0.3) -> str:
    """
    Create a BiLSTM-CRF spaCy configuration for NER training.
    
    Args:
        lang: Language code
        use_gpu: Whether to use GPU
        hidden_dim: Hidden dimension for BiLSTM
        dropout: Dropout rate
        
    Returns:
        Configuration string
    """
    config = f"""
[paths]
train = null
dev = null
vectors = null
init_tok2vec = null

[system]
gpu_allocator = {"pytorch" if use_gpu else "null"}
seed = 0

[nlp]
lang = "{lang}"
pipeline = ["tok2vec","bilstm_crf_ner"]
batch_size = 128
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {{"@tokenizers":"spacy.Tokenizer.v1"}}

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2Vec.v2"

[components.tok2vec.model.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 256
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,1000,2500,2500]
include_static_vectors = false

[components.tok2vec.model.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 256
depth = 8
window_size = 1
maxout_pieces = 3

[components.bilstm_crf_ner]
factory = "bilstm_crf_ner"
hidden_dim = {hidden_dim}
dropout = {dropout}

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2
get_length = null

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001
t = 0.0

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.001

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vectors = ${{paths.vectors}}
init_tok2vec = ${{paths.init_tok2vec}}
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""
    return config


def create_transformer_config(lang: str = "tl", transformer_name: str = "xlm-roberta-base", use_gpu: bool = False, hidden_dim: int = 256, dropout: float = 0.3) -> str:
    """
    Create a transformer + BiLSTM-CRF spaCy configuration for NER training.
    Uses TransformerListener architecture for proper fine-tuning with shared weights.
    
    Architecture:
    - Standalone Transformer component that processes docs and stores outputs in doc._.trf_data
    - BiLSTM-CRF NER component with TransformerListener tok2vec that reads from doc._.trf_data
    - Gradients flow from BiLSTM-CRF → TransformerListener → Transformer (fine-tuning)
    
    Args:
        lang: Language code
        transformer_name: HuggingFace transformer model name
        use_gpu: Whether to use GPU
        hidden_dim: Hidden dimension for BiLSTM
        dropout: Dropout rate
        
    Returns:
        Configuration string
    """
    config = f"""
[paths]
train = null
dev = null

[system]
gpu_allocator = {"pytorch" if use_gpu else "null"}
seed = 0

[nlp]
lang = "{lang}"
pipeline = ["transformer","bilstm_crf_ner_trf"]
batch_size = 128
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {{"@tokenizers":"spacy.Tokenizer.v1"}}

[components]

[components.transformer]
factory = "transformer"
max_batch_items = 4096

[components.transformer.model]
@architectures = "spacy-transformers.TransformerModel.v3"
name = "{transformer_name}"
mixed_precision = false

[components.transformer.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96

[components.transformer.model.tokenizer_config]
use_fast = true

[components.transformer.model.transformer_config]

[components.transformer.model.grad_scaler_config]

[components.bilstm_crf_ner_trf]
factory = "bilstm_crf_ner_trf"
hidden_dim = {hidden_dim}
dropout = {dropout}

[components.bilstm_crf_ner_trf.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0
upstream = "*"

[components.bilstm_crf_ner_trf.tok2vec.pooling]
@layers = "reduce_mean.v1"

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
accumulate_gradient = 3
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = 0.1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
size = 2000
buffer = 256
get_length = null

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001

[training.optimizer.learn_rate]
@schedules = "warmup_linear.v1"
warmup_steps = 250
total_steps = 20000
initial_rate = 0.00005

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""
    return config
    config = f"""
[paths]
train = null
dev = null

[system]
gpu_allocator = {"pytorch" if use_gpu else "null"}
seed = 0

[nlp]
lang = "{lang}"
pipeline = ["bilstm_crf_ner"]
batch_size = 128
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {{"@tokenizers":"spacy.Tokenizer.v1"}}

[components]

[components.bilstm_crf_ner]
factory = "bilstm_crf_ner_trf"
hidden_dim = {hidden_dim}
dropout = {dropout}
transformer_name = "{transformer_name}"

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${{paths.dev}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${{paths.train}}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${{system.seed}}
gpu_allocator = ${{system.gpu_allocator}}
dropout = 0.1
accumulate_gradient = 3
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
annotating_components = []
before_to_disk = null
before_update = null

[training.batcher]
@batchers = "spacy.batch_by_padded.v1"
discard_oversize = true
get_length = null
size = 2000
buffer = 256

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.00005

[training.score_weights]
ents_f = 1.0
ents_p = 0.0
ents_r = 0.0
ents_per_type = null

[pretraining]

[initialize]
vocab_data = null
lookups = null
before_init = null
after_init = null

[initialize.components]

[initialize.tokenizer]
"""
    return config


# ============================================================================
# MODEL INSTALLATION
# ============================================================================

def install_models():
    """Install calamanCy models for Tagalog."""
    msg.info("Installing calamanCy models...")
    
    models = [
        ("tl_calamancy_lg", "https://huggingface.co/ljvmiranda921/tl_calamancy_lg/resolve/main/tl_calamancy_lg-any-py3-none-any.whl"),
        ("tl_calamancy_trf", "https://huggingface.co/ljvmiranda921/tl_calamancy_trf/resolve/main/tl_calamancy_trf-any-py3-none-any.whl")
    ]
    
    for model_name, url in models:
        try:
            msg.info(f"Installing {model_name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", url], check=True)
            msg.good(f"Successfully installed {model_name}")
        except subprocess.CalledProcessError as e:
            msg.fail(f"Failed to install {model_name}: {e}")
            
    # Try installing calamanCy if not available
    if not CALAMANCY_AVAILABLE:
        try:
            msg.info("Installing calamanCy...")
            subprocess.run([sys.executable, "-m", "pip", "install", "calamancy==0.1.0"], check=True)
            msg.good("Successfully installed calamanCy")
        except subprocess.CalledProcessError as e:
            msg.fail(f"Failed to install calamanCy: {e}")


# ============================================================================
# DATASET PROCESSING
# ============================================================================

def process_datasets(input_path: str, output_dir: str = "corpus"):
    """
    Extract and process the TLUnified-NER dataset.
    
    Args:
        input_path: Path to the tar.gz file
        output_dir: Directory to extract files to
    """
    msg.info(f"Processing dataset from {input_path}...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(input_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
        msg.good(f"Successfully extracted dataset to {output_dir}")
        
        # List extracted files
        for file in output_path.glob("*.spacy"):
            msg.info(f"Found: {file}")
            
    except Exception as e:
        msg.fail(f"Failed to process dataset: {e}")
        raise


def convert_to_spans(input_path: str, output_path: str):
    """
    Convert IOB format to span-based format.
    
    Args:
        input_path: Input .spacy file
        output_path: Output .spacy file
    """
    msg.info(f"Converting {input_path} to spans format...")
    
    nlp = spacy.blank("tl")
    doc_bin_in = DocBin().from_disk(input_path)
    doc_bin_out = DocBin()
    
    for doc in doc_bin_in.get_docs(nlp.vocab):
        doc_bin_out.add(doc)
    
    doc_bin_out.to_disk(output_path)
    msg.good(f"Saved to {output_path}")


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_baseline(
    train_path: str = "corpus/train.spacy",
    dev_path: str = "corpus/dev.spacy",
    output_dir: str = "training/baseline",
    gpu_id: int = -1,
    config_path: Optional[str] = None
):
    """
    Train a baseline transition-based NER model without embeddings.
    
    Args:
        train_path: Path to training data (.spacy)
        dev_path: Path to development data (.spacy)
        output_dir: Output directory for trained model
        gpu_id: GPU ID (-1 for CPU)
        config_path: Optional custom config file path
    """
    msg.info("Training baseline model...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create config if not provided
    if config_path is None:
        config_path = Path(output_dir) / "config.cfg"
        config_str = create_default_config(lang="tl", use_gpu=(gpu_id >= 0))
        config_path.write_text(config_str)
        msg.info(f"Created config at {config_path}")
    
    # Train using spaCy CLI
    try:
        cmd = [
            sys.executable, "-m", "spacy", "train",
            str(config_path),
            "--output", str(output_dir),
            "--paths.train", train_path,
            "--paths.dev", dev_path,
            "--gpu-id", str(gpu_id)
        ]
        subprocess.run(cmd, check=True)
        msg.good(f"Training complete! Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        msg.fail(f"Training failed: {e}")
        raise


def train_bilstm_crf(
    train_path: str = "corpus/train.spacy",
    dev_path: str = "corpus/dev.spacy",
    output_dir: str = "training/bilstm_crf",
    gpu_id: int = -1,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    config_path: Optional[str] = None
):
    """
    Train a BiLSTM-CRF NER model with BILUO encoding.
    
    Args:
        train_path: Path to training data (.spacy)
        dev_path: Path to development data (.spacy)
        output_dir: Output directory for trained model
        gpu_id: GPU ID (-1 for CPU)
        hidden_dim: Hidden dimension for BiLSTM
        dropout: Dropout rate
        config_path: Optional custom config file path
    """
    msg.info("Training BiLSTM-CRF model with BILUO encoding...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create config if not provided
    if config_path is None:
        config_path = Path(output_dir) / "config.cfg"
        config_str = create_bilstm_crf_config(
            lang="tl",
            use_gpu=(gpu_id >= 0),
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        config_path.write_text(config_str)
        msg.info(f"Created BiLSTM-CRF config at {config_path}")
        msg.info(f"  Hidden dimension: {hidden_dim}")
        msg.info(f"  Dropout: {dropout}")
    
    # Train using spaCy CLI
    try:
        cmd = [
            sys.executable, "-m", "spacy", "train",
            str(config_path),
            "--output", str(output_dir),
            "--paths.train", train_path,
            "--paths.dev", dev_path,
            "--gpu-id", str(gpu_id),
            "--code", __file__  # Important: load custom component
        ]
        subprocess.run(cmd, check=True)
        msg.good(f"Training complete! Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        msg.fail(f"Training failed: {e}")
        raise


def train_transformer(
    train_path: str = "corpus/train.spacy",
    dev_path: str = "corpus/dev.spacy",
    output_dir: str = "training/transformer",
    transformer_name: str = "xlm-roberta-base",
    gpu_id: int = -1,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    config_path: Optional[str] = None
):
    """
    Train a transformer + BiLSTM-CRF NER model.
    
    Args:
        train_path: Path to training data
        dev_path: Path to development data
        output_dir: Output directory
        transformer_name: HuggingFace model name
        gpu_id: GPU ID
        hidden_dim: Hidden dimension for BiLSTM
        dropout: Dropout rate
        config_path: Optional custom config file path
    """
    msg.info(f"Training transformer + BiLSTM-CRF model ({transformer_name})...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create transformer config if not provided
    if config_path is None:
        config_path = Path(output_dir) / "config.cfg"
        config_str = create_transformer_config(
            lang="tl",
            transformer_name=transformer_name,
            use_gpu=(gpu_id >= 0),
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        config_path.write_text(config_str)
        msg.info(f"Created transformer + BiLSTM-CRF config at {config_path}")
        msg.info(f"  Transformer: {transformer_name}")
        msg.info(f"  Hidden dimension: {hidden_dim}")
        msg.info(f"  Dropout: {dropout}")
    
    # Train
    try:
        cmd = [
            sys.executable, "-m", "spacy", "train",
            str(config_path),
            "--output", str(output_dir),
            "--paths.train", train_path,
            "--paths.dev", dev_path,
            "--gpu-id", str(gpu_id),
            "--code", __file__  # Important: load custom component
        ]
        subprocess.run(cmd, check=True)
        msg.good(f"Training complete! Model saved to {output_dir}")
    except subprocess.CalledProcessError as e:
        msg.fail(f"Training failed: {e}")
        raise


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_model(
    model_path: str,
    test_path: str,
    output_path: Optional[str] = None,
    gpu_id: int = -1
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to trained model or model name
        test_path: Path to test data (.spacy)
        output_path: Optional path to save results JSON
        gpu_id: GPU ID
        
    Returns:
        Dictionary of evaluation metrics
    """
    msg.info(f"Evaluating model {model_path} on {test_path}...")
    
    try:
        # Load model
        nlp = spacy.load(model_path)
        
        # Load test data
        doc_bin = DocBin().from_disk(test_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        
        # Create examples
        examples = []
        for doc in docs:
            pred_doc = nlp(doc.text)
            examples.append(Example(pred_doc, doc))
        
        # Evaluate
        scores = nlp.evaluate(examples)
        
        # Print results
        msg.info("Evaluation Results:")
        msg.info(f"  Precision: {scores['ents_p']:.4f}")
        msg.info(f"  Recall: {scores['ents_r']:.4f}")
        msg.info(f"  F-score: {scores['ents_f']:.4f}")
        
        # Save to file if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(scores, f, indent=2)
            msg.good(f"Results saved to {output_path}")
        
        return scores
        
    except Exception as e:
        msg.fail(f"Evaluation failed: {e}")
        raise


def evaluate_pretrained(
    model_name: str,
    test_path: str,
    output_path: Optional[str] = None,
    gpu_id: int = -1
):
    """
    Evaluate a pre-trained calamanCy model.
    
    Args:
        model_name: Model name (e.g., "tl_calamancy_lg")
        test_path: Path to test data
        output_path: Path to save results
        gpu_id: GPU ID
    """
    return evaluate_model(model_name, test_path, output_path, gpu_id)


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def analyze_dataset(train_path: str, dev_path: str, config_path: Optional[str] = None):
    """
    Analyze dataset statistics using spaCy's debug data.
    
    Args:
        train_path: Path to training data
        dev_path: Path to development data
        config_path: Path to config file
    """
    msg.info("Analyzing dataset...")
    
    # Create temporary config if needed
    if config_path is None:
        config_path = "temp_config.cfg"
        config_str = create_default_config()
        Path(config_path).write_text(config_str)
    
    try:
        cmd = [
            sys.executable, "-m", "spacy", "debug", "data",
            config_path,
            "--paths.train", train_path,
            "--paths.dev", dev_path
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        msg.fail(f"Analysis failed: {e}")
    finally:
        # Clean up temp config
        if config_path == "temp_config.cfg" and Path(config_path).exists():
            Path(config_path).unlink()


def get_model_info(model_path: str):
    """
    Display information about a trained model.
    
    Args:
        model_path: Path to model
    """
    msg.info(f"Loading model from {model_path}...")
    
    try:
        nlp = spacy.load(model_path)
        
        msg.info("Model Information:")
        msg.info(f"  Language: {nlp.lang}")
        msg.info(f"  Pipeline: {nlp.pipe_names}")
        
        if "ner" in nlp.pipe_names:
            ner = nlp.get_pipe("ner")
            msg.info(f"  NER Labels: {ner.labels}")
        
        # Read meta.json if available
        meta_path = Path(model_path) / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            msg.info(f"  Version: {meta.get('version', 'N/A')}")
            msg.info(f"  spaCy Version: {meta.get('spacy_version', 'N/A')}")
            
            if 'performance' in meta:
                msg.info("  Performance:")
                for key, value in meta['performance'].items():
                    msg.info(f"    {key}: {value}")
                    
    except Exception as e:
        msg.fail(f"Failed to load model: {e}")


# ============================================================================
# BENCHMARK WORKFLOW
# ============================================================================

def run_benchmark(
    train_path: str = "corpus/train.spacy",
    dev_path: str = "corpus/dev.spacy",
    test_path: str = "corpus/test.spacy",
    gpu_id: int = -1
):
    """
    Run the complete benchmark workflow.
    
    Args:
        train_path: Path to training data
        dev_path: Path to development data
        test_path: Path to test data
        gpu_id: GPU ID
    """
    msg.divider("BENCHMARK WORKFLOW")
    
    # 1. Baseline
    msg.divider("1. Baseline Model")
    train_baseline(train_path, dev_path, "training/baseline", gpu_id)
    evaluate_model("training/baseline/model-best", test_path, "metrics/baseline-test.json", gpu_id)
    evaluate_model("training/baseline/model-best", dev_path, "metrics/baseline-dev.json", gpu_id)
    
    # 2. Static Vectors
    msg.divider("2. Static Vectors (calamanCy lg)")
    try:
        evaluate_pretrained("tl_calamancy_lg", test_path, "metrics/static-vectors-test.json", gpu_id)
        evaluate_pretrained("tl_calamancy_lg", dev_path, "metrics/static-vectors-dev.json", gpu_id)
    except Exception as e:
        msg.warn(f"Static vectors evaluation skipped: {e}")
    
    # 3. Transformer Monolingual
    msg.divider("3. Transformer Monolingual (calamanCy trf)")
    try:
        evaluate_pretrained("tl_calamancy_trf", test_path, "metrics/trf-monolingual-test.json", gpu_id)
        evaluate_pretrained("tl_calamancy_trf", dev_path, "metrics/trf-monolingual-dev.json", gpu_id)
    except Exception as e:
        msg.warn(f"Monolingual transformer evaluation skipped: {e}")
    
    # 4. Transformer Multilingual
    msg.divider("4. Transformer Multilingual")
    
    # XLM-RoBERTa
    msg.info("Training XLM-RoBERTa...")
    train_transformer(train_path, dev_path, "training/xlm-roberta", "xlm-roberta-base", gpu_id)
    evaluate_model("training/xlm-roberta/model-best", test_path, "metrics/trf-multilingual-xlm-test.json", gpu_id)
    evaluate_model("training/xlm-roberta/model-best", dev_path, "metrics/trf-multilingual-xlm-dev.json", gpu_id)
    
    # mBERT
    msg.info("Training mBERT...")
    train_transformer(train_path, dev_path, "training/mbert", "bert-base-multilingual-uncased", gpu_id)
    evaluate_model("training/mbert/model-best", test_path, "metrics/trf-multilingual-mbert-test.json", gpu_id)
    evaluate_model("training/mbert/model-best", dev_path, "metrics/trf-multilingual-mbert-dev.json", gpu_id)
    
    msg.good("Benchmark complete!")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Standalone Tagalog NER Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--action",
        required=True,
        choices=[
            "install", "process", "train-baseline", "train-bilstm-crf", "train-transformer",
            "evaluate", "eval-static", "eval-trf", "analyze", "info",
            "benchmark", "create-config-trf"
        ],
        help="Action to perform"
    )
    
    parser.add_argument("--input", help="Input file path")
    parser.add_argument("--train", help="Training data path")
    parser.add_argument("--dev", help="Development data path")
    parser.add_argument("--test", help="Test data path")
    parser.add_argument("--model", help="Model path or name")
    parser.add_argument("--output", help="Output path/directory")
    parser.add_argument("--transformer", default="xlm-roberta-base", help="Transformer model name")
    parser.add_argument("--gpu-id", type=int, default=-1, help="GPU ID (-1 for CPU)")
    parser.add_argument("--config", help="Config file path")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension for BiLSTM-CRF")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate for BiLSTM-CRF")
    
    args = parser.parse_args()
    
    try:
        if args.action == "install":
            install_models()
            
        elif args.action == "process":
            if not args.input:
                msg.fail("--input required for process action")
                return
            process_datasets(args.input, args.output or "corpus")
            
        elif args.action == "train-baseline":
            if not args.train or not args.dev:
                msg.fail("--train and --dev required for training")
                return
            train_baseline(
                args.train, args.dev,
                args.output or "training/baseline",
                args.gpu_id, args.config
            )
            
        elif args.action == "train-bilstm-crf":
            if not args.train or not args.dev:
                msg.fail("--train and --dev required for training")
                return
            if not TORCH_AVAILABLE:
                msg.fail("PyTorch is required for BiLSTM-CRF training. Please install torch and torchcrf.")
                return
            train_bilstm_crf(
                args.train, args.dev,
                args.output or "training/bilstm_crf",
                args.gpu_id,
                args.hidden_dim,
                args.dropout,
                args.config
            )
            
        elif args.action == "train-transformer":
            if not args.train or not args.dev:
                msg.fail("--train and --dev required for training")
                return
            if not TORCH_AVAILABLE:
                msg.fail("PyTorch is required for transformer + BiLSTM-CRF training. Please install torch and torchcrf.")
                return
            train_transformer(
                args.train, args.dev,
                args.output or f"training/{args.transformer}",
                args.transformer, args.gpu_id,
                args.hidden_dim,
                args.dropout,
                args.config
            )
            
        elif args.action == "evaluate":
            if not args.model or not args.test:
                msg.fail("--model and --test required for evaluation")
                return
            evaluate_model(args.model, args.test, args.output, args.gpu_id)
            
        elif args.action == "eval-static":
            if not args.test:
                msg.fail("--test required for evaluation")
                return
            evaluate_pretrained("tl_calamancy_lg", args.test, args.output, args.gpu_id)
            
        elif args.action == "eval-trf":
            if not args.test:
                msg.fail("--test required for evaluation")
                return
            evaluate_pretrained("tl_calamancy_trf", args.test, args.output, args.gpu_id)
            
        elif args.action == "analyze":
            if not args.train or not args.dev:
                msg.fail("--train and --dev required for analysis")
                return
            analyze_dataset(args.train, args.dev, args.config)
            
        elif args.action == "info":
            if not args.model:
                msg.fail("--model required for info action")
                return
            get_model_info(args.model)
            
        elif args.action == "benchmark":
            train = args.train or "corpus/train.spacy"
            dev = args.dev or "corpus/dev.spacy"
            test = args.test or "corpus/test.spacy"
            run_benchmark(train, dev, test, args.gpu_id)
        
        elif args.action == "create-config-trf":
            if not args.output:
                msg.fail("--output required for create-config-trf action")
                return
            
            # Generate transformer config
            use_gpu = args.gpu_id >= 0
            config_str = create_transformer_config(
                lang="tl",
                transformer_name=args.transformer,
                use_gpu=use_gpu,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout
            )
            
            # Save config to file
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(config_str, encoding="utf-8")
            
            msg.good(f"Created transformer config at {output_path}")
            msg.info(f"  Transformer: {args.transformer}")
            msg.info(f"  Hidden dim: {args.hidden_dim}")
            msg.info(f"  Dropout: {args.dropout}")
            msg.info(f"  GPU: {use_gpu}")
            
    except Exception as e:
        msg.fail(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
