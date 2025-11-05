"""
Debug script to check BILUO label generation
"""
import spacy
from spacy.training import Corpus

# Load corpus
corpus = Corpus("corpus/train.spacy")

# Get first few examples
examples = list(corpus(spacy.blank("tl")))[:5]

print(f"Total examples loaded: {len(examples)}")
print("\n" + "="*80)

for i, eg in enumerate(examples[:5]):
    print(f"\nExample {i+1}:")
    print(f"Text: {eg.reference.text[:100]}...")
    print(f"Entities: {[(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in eg.reference.ents]}")
    print(f"Number of tokens: {len(eg.reference)}")
    
    # Check what labels would be
    entity_labels = set()
    for ent in eg.reference.ents:
        entity_labels.add(ent.label_)
    
    print(f"Unique entity labels: {entity_labels}")
    
print("\n" + "="*80)
print("\nAll unique entity labels across all examples:")
all_labels = set()
for eg in examples:
    for ent in eg.reference.ents:
        all_labels.add(ent.label_)
print(f"Labels: {sorted(all_labels)}")

# Manually compute BILUO
print("\n" + "="*80)
print("\nBILUO mapping that would be created:")
biluo_labels = ['O']
for label in sorted(all_labels):
    biluo_labels.extend([f'B-{label}', f'I-{label}', f'L-{label}', f'U-{label}'])
    
for idx, label in enumerate(biluo_labels):
    print(f"  {idx}: {label}")

print(f"\nTotal BILUO labels: {len(biluo_labels)}")
