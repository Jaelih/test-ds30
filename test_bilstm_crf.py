"""
Test script for BiLSTM-CRF component integration.

This script verifies that the custom BiLSTM-CRF component is properly registered
and can be used with spaCy.
"""

import spacy
from spacy.tokens import Doc, DocBin
from spacy.training import Example
from pathlib import Path
import sys

# Import the standalone script to register the component
import standalone_ner_pipeline

def test_component_registration():
    """Test that the BiLSTM-CRF component is registered."""
    print("Testing component registration...")
    
    # Create a blank Tagalog pipeline
    nlp = spacy.blank("tl")
    
    # Check if factory is registered
    if "bilstm_crf_ner" in nlp.factory_names:
        print("✓ bilstm_crf_ner factory is registered")
    else:
        print("✗ bilstm_crf_ner factory is NOT registered")
        return False
    
    return True


def test_component_creation():
    """Test creating a pipeline with BiLSTM-CRF."""
    print("\nTesting component creation...")
    
    try:
        # Create pipeline
        nlp = spacy.blank("tl")
        
        # Add tok2vec
        tok2vec = nlp.add_pipe("tok2vec")
        print("✓ Added tok2vec component")
        
        # Add BiLSTM-CRF
        bilstm_crf = nlp.add_pipe("bilstm_crf_ner", config={
            "hidden_dim": 128,
            "dropout": 0.3,
            "labels": ["PER", "ORG", "LOC"]
        })
        print("✓ Added bilstm_crf_ner component")
        
        print(f"  Pipeline: {nlp.pipe_names}")
        print(f"  Labels: {bilstm_crf.labels}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error creating component: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction():
    """Test making predictions with the component."""
    print("\nTesting prediction...")
    
    try:
        # Create pipeline
        nlp = spacy.blank("tl")
        tok2vec = nlp.add_pipe("tok2vec")
        bilstm_crf = nlp.add_pipe("bilstm_crf_ner", config={
            "hidden_dim": 128,
            "dropout": 0.3,
            "labels": ["PER", "ORG", "LOC"]
        })
        
        # Initialize with dummy data
        def get_examples():
            doc = nlp.make_doc("Juan dela Cruz ay nag-trabaho sa Manila.")
            predicted = doc.copy()
            reference = doc.copy()
            return [Example(predicted, reference)]
        
        # Initialize tok2vec first (required for bilstm_crf)
        print("  Initializing tok2vec...")
        tok2vec.initialize(get_examples, nlp=nlp)
        
        print("  Initializing bilstm_crf_ner...")
        bilstm_crf.initialize(get_examples, nlp=nlp, labels=["PER", "ORG", "LOC"])
        print("✓ Component initialized")
        
        # Make prediction
        doc = nlp("Maria Santos ay taga-Quezon City.")
        print(f"✓ Prediction completed")
        print(f"  Input: {doc.text}")
        print(f"  Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_biluo_conversion():
    """Test BILUO tag conversion."""
    print("\nTesting BILUO conversion...")
    
    try:
        # Create pipeline
        nlp = spacy.blank("tl")
        tok2vec = nlp.add_pipe("tok2vec")
        bilstm_crf = nlp.add_pipe("bilstm_crf_ner", config={
            "hidden_dim": 128,
            "dropout": 0.3,
            "labels": ["PER", "ORG", "LOC"]
        })
        
        # Initialize components
        def get_examples():
            doc = nlp.make_doc("Juan dela Cruz ay nag-trabaho sa Manila.")
            predicted = doc.copy()
            reference = doc.copy()
            return [Example(predicted, reference)]
        
        tok2vec.initialize(get_examples, nlp=nlp)
        bilstm_crf.initialize(get_examples, nlp=nlp, labels=["PER", "ORG", "LOC"])
        
        # Test document with entities (process through nlp to set vocab)
        doc = nlp.make_doc("Juan dela Cruz works at Google Philippines")
        
        # Manually set entities with proper Span objects
        from spacy.tokens import Span
        spans = []
        # Juan dela Cruz (tokens 0-2)
        spans.append(Span(doc, 0, 3, label="PER"))
        # Google Philippines (tokens 5-6)  
        spans.append(Span(doc, 5, 7, label="ORG"))
        doc.ents = spans
        
        # Convert to BILUO
        biluo_tags = bilstm_crf._spans_to_biluo(doc)
        print(f"✓ BILUO conversion completed")
        print(f"  Tokens: {[token.text for token in doc]}")
        print(f"  BILUO indices: {biluo_tags}")
        print(f"  BILUO tags: {[bilstm_crf._idx_to_label.get(idx, 'O') for idx in biluo_tags]}")
        
        # Convert back to spans
        spans = bilstm_crf._biluo_to_spans(doc, biluo_tags)
        print(f"  Recovered spans: {[(doc[s:e].text, label) for s, e, label in spans]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during BILUO conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("BiLSTM-CRF Component Integration Tests")
    print("=" * 60)
    
    tests = [
        test_component_registration,
        test_component_creation,
        test_prediction,
        test_biluo_conversion,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
