"""
Transformer-based NER Pipeline for Tagalog

This module contains BiLSTM, BiLSTM-CRF, and CRF implementations that use transformer fine-tuning
via spaCy's TransformerListener architecture for proper gradient flow.

Components:
- BiLSTMNER + TransformerListener: BiLSTM with fine-tuned transformer
- BiLSTMCRFNER + TransformerListener: BiLSTM-CRF with fine-tuned transformer
- CRFNER + TransformerListener: CRF with fine-tuned transformer

Architecture:
- Upstream Transformer component processes documents
- Downstream components use TransformerListener to receive embeddings
- Gradients flow back: Model → TransformerListener → Transformer (fine-tuning)

Usage:
    # Train BiLSTM with XLM-RoBERTa
    python transformer_pipeline.py --action train --model bilstm --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_xlmr --transformer xlm-roberta-base
    
    # Train BiLSTM-CRF with XLM-RoBERTa
    python transformer_pipeline.py --action train --model bilstm-crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/bilstm_crf_xlmr --transformer xlm-roberta-base
    
    # Train CRF with mBERT
    python transformer_pipeline.py --action train --model crf --train corpus/train.spacy --dev corpus/dev.spacy --output training/crf_mbert --transformer bert-base-multilingual-cased
    
    # Evaluate model

    python transformer_pipeline.py --action evaluate --model training/bilstm_xlmr/model-best --test corpus/test.spacy
    
    # Generate config
    python transformer_pipeline.py --action create-config --model bilstm-crf --transformer xlm-roberta-base --output configs/bilstm_crf_xlmr.cfg
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from wasabi import msg

# Import shared components
try:
    from transformer_components import TORCH_AVAILABLE
except ImportError:
    msg.fail("Could not import from transformer_components.py")
    msg.text("Make sure transformer_components.py is in the same directory")
    sys.exit(1)


def create_config(
    model_type: str,
    transformer_name: str = "xlm-roberta-base",
    lang: str = "tl",
    use_gpu: bool = False,
    hidden_dim: int = 256,
    dropout: float = 0.3
) -> str:
    """
    Create spaCy config for transformer-based models with TransformerListener.
    
    Args:
        model_type: One of 'bilstm', 'bilstm-crf', 'crf'
        transformer_name: HuggingFace transformer model name
        lang: Language code
        use_gpu: Whether to use GPU
        hidden_dim: Hidden dimension for LSTM/CRF
        dropout: Dropout rate
    
    Returns:
        Configuration string
    """
    # Determine factory name
    if model_type == "bilstm":
        factory = "bilstm_ner_trf"
        component_name = "bilstm_ner_trf"
    elif model_type == "bilstm-crf":
        factory = "bilstm_crf_ner_trf"
        component_name = "bilstm_crf_ner_trf"
    elif model_type == "crf":
        factory = "crf_ner_trf"
        component_name = "crf_ner_trf"
        hidden_dim = 0
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    config = f"""
[paths]
train = null
dev = null

[system]
gpu_allocator = {"pytorch" if use_gpu else "null"}
seed = 0

[nlp]
lang = "{lang}"
pipeline = ["transformer","{component_name}"]
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

[components.{component_name}]
factory = "{factory}"
{"" if model_type == "crf" else ("hidden_dim = " + str(hidden_dim))}
dropout = {dropout}

[components.{component_name}.tok2vec]
@architectures = "spacy-transformers.TransformerListener.v1"
grad_factor = 1.0
upstream = "*"

[components.{component_name}.tok2vec.pooling]
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
eval_frequency = 5
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


def train_model(
    model_type: str,
    train_path: str,
    dev_path: str,
    output_dir: str,
    transformer_name: str = "xlm-roberta-base",
    gpu_id: int = -1,
    hidden_dim: int = 256,
    dropout: float = 0.3,
    config_path: Optional[str] = None
):
    """Train a transformer-based model."""
    if not TORCH_AVAILABLE:
        msg.fail("PyTorch with torchcrf must be installed before training transformer models.")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate config if not provided
    if config_path is None:
        use_gpu = gpu_id >= 0
        config = create_config(
            model_type,
            transformer_name=transformer_name,
            use_gpu=use_gpu,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        config_path = output_path / "config.cfg"
        
        with open(config_path, "w") as f:
            f.write(config)
        
        msg.good(f"Created config at {config_path}")
        msg.info(f"  Model: {model_type} + {transformer_name}")
        msg.info(f"  Hidden dimension: {hidden_dim}")
        msg.info(f"  Dropout: {dropout}")
        msg.info(f"  Architecture: TransformerListener (fine-tuning enabled)")
    
    # Build training command
    code_path = Path(__file__).with_name("transformer_components.py")

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", train_path,
        "--paths.dev", dev_path,
        "--gpu-id", str(gpu_id),
        "--code", str(code_path)
    ]
    
    msg.info(f"Training {model_type} + transformer model...")
    msg.text(f"Command: {' '.join(cmd)}")
    
    # Run training
    subprocess.run(cmd, check=True)
    msg.good(f"Training complete! Model saved to {output_dir}")


def evaluate_model(model_path: str, test_path: str, gpu_id: int = -1, output_path: Optional[str] = None):
    """Evaluate a trained model."""
    if not TORCH_AVAILABLE:
        msg.fail("PyTorch with torchcrf must be installed before evaluating transformer models.")
        sys.exit(1)

    code_path = Path(__file__).with_name("transformer_components.py")

    cmd = [
        sys.executable, "-m", "spacy", "evaluate",
        model_path,
        test_path,
        "--gpu-id", str(gpu_id),
        "--code", str(code_path)
    ]
    
    if output_path:
        cmd.extend(["--output", output_path])
    
    msg.info("Evaluating model...")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Transformer-based NER Pipeline")
    parser.add_argument(
        "--action",
        choices=["train", "evaluate", "create-config"],
        required=True,
        help="Action to perform"
    )
    parser.add_argument(
        "--model",
        choices=["bilstm", "bilstm-crf", "crf"],
        # required=True,
        help="Model architecture"
    )
    parser.add_argument(
        "--transformer",
        default="xlm-roberta-base",
        help="HuggingFace transformer model name. Choose from 'xlm-roberta-base', 'bert-base-multilingual-cased', 'jcblaise/roberta-tagalog-base'"
    )
    parser.add_argument("--train", help="Path to training data (.spacy)")
    parser.add_argument("--dev", help="Path to development data (.spacy)")
    parser.add_argument("--test", help="Path to test data (.spacy)")
    parser.add_argument("--model-path", help="Path to trained model (for evaluation)")
    parser.add_argument("--output", required=True, help="Output directory or file path")
    parser.add_argument("--gpu-id", type=int, default=-1, help="GPU ID (-1 for CPU)")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--config", help="Path to existing config file")
    
    args = parser.parse_args()
    
    if args.action == "train":
        if not args.train or not args.dev or not args.model or not args.output:
            msg.fail("--train, --dev, --model, and --output are required for training")
            sys.exit(1)
        
        train_model(
            model_type=args.model,
            train_path=args.train,
            dev_path=args.dev,
            output_dir=args.output,
            transformer_name=args.transformer,
            gpu_id=args.gpu_id,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            config_path=args.config
        )
    
    elif args.action == "evaluate":
        if not args.model_path or not args.test:
            msg.fail("--model-path and --test are required for evaluation")
            sys.exit(1)
        
        evaluate_model(
            model_path=args.model_path,
            test_path=args.test,
            gpu_id=args.gpu_id,
            output_path=args.output
        )
    
    elif args.action == "create-config":
        use_gpu = args.gpu_id >= 0
        config = create_config(
            model_type=args.model,
            transformer_name=args.transformer,
            use_gpu=use_gpu,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
        
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            f.write(config)
        
        msg.good(f"Config saved to {output_file}")


if __name__ == "__main__":
    main()
