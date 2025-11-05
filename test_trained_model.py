"""
Test what the trained BiLSTM-CRF model actually predicts
"""
import spacy

# Import to register the component
import standalone_ner_pipeline

# Load the trained model
nlp = spacy.load("training/bilstm_crf/model-last")

# Test sentences with known entities
test_texts = [
    "Juan dela Cruz works at Google Philippines in Manila.",
    "Si Maria Santos ay nakatira sa Quezon City.",
    "Ang MMDA ay nag-organisa ng meeting sa Makati."
]

print("Testing trained BiLSTM-CRF model:\n")
print("="*80)

for text in test_texts:
    doc = nlp(text)
    print(f"\nText: {text}")
    print(f"Entities found: {len(doc.ents)}")
    if doc.ents:
        for ent in doc.ents:
            print(f"  - {ent.text} ({ent.label_})")
    else:
        print("  (No entities detected)")
    
    # Check the bilstm_crf component directly
    bilstm_crf = nlp.get_pipe("bilstm_crf_ner")
    print(f"  Label map: {len(bilstm_crf._label_map)} labels")
    
    # Get predictions
    predictions = bilstm_crf.predict([doc])
    if predictions and len(predictions) > 0:
        pred_tags = predictions[0]
        print(f"  Predicted tags: {pred_tags[:20]}...")  # First 20
        
        # Count non-O predictions
        non_o_count = sum(1 for tag in pred_tags if tag != 0)
        print(f"  Non-O predictions: {non_o_count}/{len(pred_tags)}")
        
        # Show unique predicted tags
        unique_tags = set(pred_tags)
        unique_labels = [bilstm_crf._idx_to_label.get(tag, f"UNKNOWN-{tag}") for tag in unique_tags]
        print(f"  Unique tags predicted: {unique_labels}")

print("\n" + "="*80)
