"""
Check if corpus data has entities
"""
import spacy
from spacy.tokens import DocBin

# Load training data
train_path = "corpus/train.spacy"
docbin = DocBin().from_disk(train_path)

nlp = spacy.blank("tl")
docs = list(docbin.get_docs(nlp.vocab))

print(f"Total documents: {len(docs)}")
print(f"\nFirst 5 documents:")

total_ents = 0
for i, doc in enumerate(docs[:5]):
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    total_ents += len(doc.ents)
    print(f"\nDoc {i+1}:")
    print(f"  Text: {doc.text[:80]}...")
    print(f"  Entities ({len(doc.ents)}): {ents[:3]}")

print(f"\nTotal entities in first 5 docs: {total_ents}")

# Check all docs
all_ents = sum(len(doc.ents) for doc in docs)
print(f"Total entities in all {len(docs)} docs: {all_ents}")
print(f"Average entities per doc: {all_ents / len(docs):.2f}")
