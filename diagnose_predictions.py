"""
Diagnostic script to understand why F1 scores are 0.00 despite decreasing loss.

This script will:
1. Load the trained model
2. Make predictions on dev set
3. Compare predictions vs gold labels
4. Analyze what's going wrong
"""
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from collections import Counter
from wasabi import msg
import standalone_ner_pipeline_iob  # Register IOB2 component

# Load trained IOB2 model
msg.info("Loading IOB2 model...")
nlp = spacy.load("training/bilstm_crf_iob/model-last")

# Load dev data
msg.info("Loading dev data...")
doc_bin = DocBin().from_disk("corpus/dev.spacy")
docs_gold = list(doc_bin.get_docs(nlp.vocab))

msg.divider("ANALYSIS")

# Sample a few examples
num_samples = 10
msg.info(f"Analyzing {num_samples} examples from dev set...\n")

total_gold_ents = 0
total_pred_ents = 0
total_correct = 0
predicted_tags_counter = Counter()
gold_tags_counter = Counter()

for i, gold_doc in enumerate(docs_gold[:num_samples]):
    # Make prediction
    pred_doc = nlp(gold_doc.text)
    
    # Count entities
    gold_ents = list(gold_doc.ents)
    pred_ents = list(pred_doc.ents)
    
    total_gold_ents += len(gold_ents)
    total_pred_ents += len(pred_ents)
    
    # Check for exact matches
    correct = 0
    for pred_ent in pred_ents:
        for gold_ent in gold_ents:
            if (pred_ent.start == gold_ent.start and 
                pred_ent.end == gold_ent.end and
                pred_ent.label_ == gold_ent.label_):
                correct += 1
                break
    total_correct += correct
    
    # Print example
    msg.info(f"Example {i+1}:")
    msg.info(f"  Text: {gold_doc.text[:80]}...")
    msg.info(f"  Gold entities: {len(gold_ents)}")
    if gold_ents:
        for ent in gold_ents[:3]:
            msg.info(f"    - {ent.text} ({ent.label_}) [{ent.start}:{ent.end}]")
    
    msg.info(f"  Predicted entities: {len(pred_ents)}")
    if pred_ents:
        for ent in pred_ents[:3]:
            msg.info(f"    - {ent.text} ({ent.label_}) [{ent.start}:{ent.end}]")
    else:
        msg.warn("    (No entities predicted)")
    
    msg.info(f"  Correct: {correct}\n")
    
    # Get component predictions (tags)
    bilstm_crf = nlp.get_pipe("bilstm_crf_ner_iob")
    pred_tags = bilstm_crf.predict([pred_doc])[0]
    gold_tags = bilstm_crf._spans_to_iob(gold_doc)
    
    # Count tag distribution
    for tag_idx in pred_tags:
        tag_name = bilstm_crf._idx_to_label.get(tag_idx, f"UNK-{tag_idx}")
        predicted_tags_counter[tag_name] += 1
    
    for tag_idx in gold_tags:
        tag_name = bilstm_crf._idx_to_label.get(tag_idx, f"UNK-{tag_idx}")
        gold_tags_counter[tag_name] += 1

msg.divider("SUMMARY")
msg.info(f"Total gold entities: {total_gold_ents}")
msg.info(f"Total predicted entities: {total_pred_ents}")
msg.info(f"Total correct (exact match): {total_correct}")

if total_gold_ents > 0:
    recall = total_correct / total_gold_ents
    msg.info(f"Approximate Recall: {recall:.2%}")

if total_pred_ents > 0:
    precision = total_correct / total_pred_ents
    msg.info(f"Approximate Precision: {precision:.2%}")
else:
    msg.warn("No entities predicted at all!")

msg.divider("TAG DISTRIBUTION")
msg.info("Predicted tags:")
for tag, count in predicted_tags_counter.most_common():
    msg.info(f"  {tag}: {count}")

msg.info("\nGold tags:")
for tag, count in gold_tags_counter.most_common():
    msg.info(f"  {tag}: {count}")

# Check if model is just predicting all O
if predicted_tags_counter.get('O', 0) == sum(predicted_tags_counter.values()):
    msg.fail("MODEL IS PREDICTING ALL 'O' TAGS - NOT LEARNING ENTITIES!")
elif predicted_tags_counter.get('O', 0) > 0.95 * sum(predicted_tags_counter.values()):
    msg.warn("MODEL IS PREDICTING MOSTLY 'O' TAGS")
else:
    msg.good("Model is predicting diverse tags")
