"""
Debug script to test BiLSTM-CRF training locally
"""
import spacy
from spacy.training import Example
import torch

# Import to register the component
import standalone_ner_pipeline

# Create pipeline
nlp = spacy.blank("tl")
tok2vec = nlp.add_pipe("tok2vec")
bilstm_crf = nlp.add_pipe("bilstm_crf_ner", config={
    "hidden_dim": 128,
    "dropout": 0.3,
    "labels": ["PER", "ORG", "LOC"]
})

# Create examples
def get_examples():
    examples = []
    
    # Example 1
    text1 = "Juan dela Cruz works at Google Philippines"
    doc1 = nlp.make_doc(text1)
    gold1 = {"entities": [(0, 15, "PER"), (26, 45, "ORG")]}
    examples.append(Example.from_dict(doc1, gold1))
    
    # Example 2
    text2 = "Maria Santos lives in Manila"
    doc2 = nlp.make_doc(text2)
    gold2 = {"entities": [(0, 13, "PER"), (23, 29, "LOC")]}
    examples.append(Example.from_dict(doc2, gold2))
    
    return examples

# Initialize
examples = get_examples()
tok2vec.initialize(lambda: examples, nlp=nlp)
bilstm_crf.initialize(lambda: examples, nlp=nlp, labels=["PER", "ORG", "LOC"])

print("Initialized!")
print(f"Labels: {bilstm_crf.labels}")
print(f"BILUO label map: {bilstm_crf._label_map}")

# Try one update
print("\nBefore update:")
losses = {}
bilstm_crf.update(examples, losses=losses)
print(f"Losses: {losses}")

# Check if loss is non-zero
if losses.get("bilstm_crf_ner", 0.0) > 0:
    print("✓ Loss is non-zero! Training is working.")
else:
    print("✗ Loss is zero. Something is wrong.")
    
    # Debug: Check what's being passed
    print("\nDebug info:")
    for i, eg in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"  Text: {eg.reference.text}")
        print(f"  Entities: {[(ent.text, ent.label_) for ent in eg.reference.ents]}")
        
        # Check BILUO tags
        biluo_tags = bilstm_crf._spans_to_biluo(eg.reference)
        print(f"  BILUO tags: {biluo_tags}")
        print(f"  Unique tags: {set(biluo_tags)}")
