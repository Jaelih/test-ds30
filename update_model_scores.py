"""
Update meta.json files with correct evaluation scores.

This script loads trained models, evaluates them on test/dev data,
and updates the meta.json files with actual performance metrics.

USAGE:
------
# Evaluate and update a single model
python update_model_scores.py --model training/bilstm_crf_iob/model-best --data corpus/dev.spacy

# Evaluate multiple models
python update_model_scores.py --model training/bilstm_crf/model-best --data corpus/dev.spacy
python update_model_scores.py --model training/baseline/model-best --data corpus/test.spacy

# Batch update all models
python update_model_scores.py --batch --data corpus/dev.spacy
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, Any
import spacy
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from spacy.training import Example
from wasabi import msg

# Import custom components to register them
try:
    import standalone_ner_pipeline  # Registers bilstm_crf_ner
    import standalone_ner_pipeline_iob  # Registers bilstm_crf_ner_iob
    msg.good("Custom components registered")
except ImportError as e:
    msg.warn(f"Could not import custom components: {e}")


def evaluate_model(model_path: str, data_path: str) -> Dict[str, Any]:
    """
    Evaluate a model and return metrics.
    
    Args:
        model_path: Path to trained model directory
        data_path: Path to evaluation data (.spacy file)
        
    Returns:
        Dictionary with evaluation scores
    """
    msg.info(f"Loading model from {model_path}...")
    
    # Load model - custom components already registered
    nlp = spacy.load(model_path)
    
    msg.info(f"Pipeline: {nlp.pipe_names}")
    msg.info(f"Loading evaluation data from {data_path}...")
    doc_bin = DocBin().from_disk(data_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    
    msg.info(f"Evaluating on {len(docs)} documents...")
    
    # Create examples
    examples = []
    for gold_doc in docs:
        pred_doc = nlp(gold_doc.text)
        examples.append(Example(pred_doc, gold_doc))
    
    # Score
    scorer = Scorer()
    scores = scorer.score(examples)
    
    # Extract relevant metrics
    metrics = {
        "ents_f": scores.get("ents_f", 0.0),
        "ents_p": scores.get("ents_p", 0.0),
        "ents_r": scores.get("ents_r", 0.0),
        "ents_per_type": scores.get("ents_per_type", {}),
    }
    
    msg.info(f"Results:")
    msg.info(f"  F1:        {metrics['ents_f']:.4f}")
    msg.info(f"  Precision: {metrics['ents_p']:.4f}")
    msg.info(f"  Recall:    {metrics['ents_r']:.4f}")
    
    if metrics["ents_per_type"]:
        msg.info("  Per-type scores:")
        for ent_type, type_scores in metrics["ents_per_type"].items():
            msg.info(f"    {ent_type}: P={type_scores.get('p', 0):.4f}, R={type_scores.get('r', 0):.4f}, F={type_scores.get('f', 0):.4f}")
    
    return metrics


def update_meta_json(model_path: str, metrics: Dict[str, Any], backup: bool = True):
    """
    Update the meta.json file with correct scores.
    
    Args:
        model_path: Path to model directory
        metrics: Dictionary of evaluation metrics
        backup: Whether to create a backup of original meta.json
    """
    meta_path = Path(model_path) / "meta.json"
    
    if not meta_path.exists():
        msg.fail(f"meta.json not found at {meta_path}")
        return
    
    # Load existing meta.json
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Backup if requested
    if backup:
        backup_path = meta_path.with_suffix(".json.backup")
        with open(backup_path, "w") as f:
            json.dump(meta, f, indent=2)
        msg.info(f"Created backup at {backup_path}")
    
    # Update performance metrics
    if "performance" not in meta:
        meta["performance"] = {}
    
    meta["performance"]["ents_f"] = metrics["ents_f"]
    meta["performance"]["ents_p"] = metrics["ents_p"]
    meta["performance"]["ents_r"] = metrics["ents_r"]
    
    # Add per-type scores if available
    if metrics.get("ents_per_type"):
        meta["performance"]["ents_per_type"] = metrics["ents_per_type"]
    
    # Write updated meta.json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    msg.good(f"Updated {meta_path}")


def create_scores_json(model_path: str, metrics: Dict[str, Any], data_name: str = "dev"):
    """
    Create a separate JSON file with scores (alternative to updating meta.json).
    
    Args:
        model_path: Path to model directory
        metrics: Dictionary of evaluation metrics
        data_name: Name of dataset (e.g., "dev", "test")
    """
    model_dir = Path(model_path)
    output_path = model_dir / f"scores_{data_name}.json"
    
    # Load meta.json for context
    meta_path = model_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        
        scores_data = {
            "model_name": meta.get("name", "unknown"),
            "model_version": meta.get("version", "0.0.0"),
            "pipeline": meta.get("pipeline", []),
            "labels": meta.get("labels", {}),
            "dataset": data_name,
            "evaluation_scores": metrics
        }
    else:
        scores_data = {
            "dataset": data_name,
            "evaluation_scores": metrics
        }
    
    with open(output_path, "w") as f:
        json.dump(scores_data, f, indent=2)
    
    msg.good(f"Created scores file at {output_path}")


def batch_update(data_path: str, models_dir: str = "training"):
    """
    Update scores for all models in the training directory.
    
    Args:
        data_path: Path to evaluation data
        models_dir: Directory containing trained models
    """
    models_dir = Path(models_dir)
    
    # Find all model-best directories
    model_paths = list(models_dir.glob("*/model-best"))
    
    if not model_paths:
        msg.warn(f"No model-best directories found in {models_dir}")
        return
    
    msg.info(f"Found {len(model_paths)} models to evaluate")
    
    for model_path in model_paths:
        msg.divider(f"Evaluating {model_path.parent.name}")
        try:
            metrics = evaluate_model(str(model_path), data_path)
            update_meta_json(str(model_path), metrics)
            create_scores_json(str(model_path), metrics, "dev")
        except Exception as e:
            msg.fail(f"Failed to evaluate {model_path}: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Update model meta.json files with correct evaluation scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model",
        help="Path to model directory (e.g., training/bilstm_crf_iob/model-best)"
    )
    parser.add_argument(
        "--data",
        help="Path to evaluation data (.spacy file)"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Update all models in training directory"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup of original meta.json"
    )
    parser.add_argument(
        "--create-separate",
        action="store_true",
        help="Create separate scores JSON instead of updating meta.json"
    )
    parser.add_argument(
        "--models-dir",
        default="training",
        help="Directory containing models (for --batch mode)"
    )
    
    args = parser.parse_args()
    
    if not args.data:
        msg.fail("--data is required")
        return
    
    if not Path(args.data).exists():
        msg.fail(f"Data file not found: {args.data}")
        return
    
    try:
        if args.batch:
            batch_update(args.data, args.models_dir)
        elif args.model:
            if not Path(args.model).exists():
                msg.fail(f"Model not found: {args.model}")
                return
            
            msg.divider("Evaluating Model")
            metrics = evaluate_model(args.model, args.data)
            
            if args.create_separate:
                create_scores_json(args.model, metrics, Path(args.data).stem)
            else:
                update_meta_json(args.model, metrics, backup=not args.no_backup)
                create_scores_json(args.model, metrics, Path(args.data).stem)
        else:
            msg.fail("Either --model or --batch is required")
            
    except Exception as e:
        msg.fail(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
