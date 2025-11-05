"""
Quick test to see if BiLSTM-CRF is learning
Train for just 1000 steps and check if F1 improves
"""
import subprocess
import sys

cmd = [
    sys.executable,
    "standalone_ner_pipeline.py",
    "--action", "train-bilstm-crf",
    "--train", "corpus/train.spacy",
    "--dev", "corpus/dev.spacy",
    "--output", "training/bilstm_crf_test",
    "--gpu-id", "0"
]

print("=" * 70)
print("TESTING IF BiLSTM-CRF IS LEARNING")
print("=" * 70)
print("\nTraining for 1000 steps...")
print("Watch for F1 scores (ENTS_F column) - they should increase if learning")
print("\nIf F1 stays at 0.00, model is NOT learning")
print("If F1 increases (even to 0.01, 0.05, etc.), model IS learning!\n")
print("=" * 70)
print()

try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"\nTraining failed with error: {e}")
    sys.exit(1)
