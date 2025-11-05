#!/usr/bin/env python3
"""
Standalone spaCy V3 Training Script
Supports: model creation, training, evaluation, and packaging
"""



import argparse
import spacy
from spacy import cli as spacy_cli
from pathlib import Path
import sys
import subprocess


def create_config(output_path: str, lang: str = "tl", pipeline: str = "ner"):
  """Create a default config file"""
  print(f"Creating config file at {output_path}...")
def create_config(output_path: str, lang: str = "tl", pipeline: str = "ner"):
  """Create a default config file"""
  print(f"Creating config file at {output_path}...")
  subprocess.run([
    sys.executable, "-m", "spacy", "init", "config",
    output_path, "--lang", lang, "--pipeline", pipeline,
    "--optimize", "efficiency"
  ], check=True)
  print("Config file created successfully!")
  """Initialize a model from config"""
  print(f"Initializing model from {config_path}...")
def initialize_model(config_path: str, output_path: str):
  """Initialize a model from config"""
  print(f"Initializing model from {config_path}...")
  subprocess.run([
    sys.executable, "-m", "spacy", "init", "fill-config",
    config_path, output_path
  ], check=True)
  print(f"Model initialized at {output_path}")
  print(f"Output directory: {output_path}")
  
def train_model(config_path: str, output_path: str, overrides: dict = None):
  """Train a spaCy model"""
  print(f"Training model with config: {config_path}")
  print(f"Output directory: {output_path}")
  
  cmd = [sys.executable, "-m", "spacy", "train", config_path, "--output", output_path]
  if overrides:
    for key, value in overrides.items():
      cmd.extend(["--override", f"{key}={value}"])
  
  subprocess.run(cmd, check=True)
  print("Training completed!")
  # print(f"Test data: {data_path}")  # Removed or commented out as data_path is undefined
def evaluate_model(model_path: str, data_path: str, output_path: str = None):
  """Evaluate a trained model"""
  print(f"Evaluating model: {model_path}")
  print(f"Test data: {data_path}")
  
  cmd = [sys.executable, "-m", "spacy", "evaluate", model_path, data_path]
  if output_path:
    cmd.extend(["--output", output_path])
  
  subprocess.run(cmd, check=True)
  print("\nEvaluation completed!")
  # print(f"Packaging model from {input_dir}...")
def package_model(input_dir: str, output_dir: str, name: str, version: str):
  """Package a trained model"""
  print(f"Packaging model from {input_dir}...")
  
  subprocess.run([
    sys.executable, "-m", "spacy", "package",
    input_dir, output_dir,
    "--name", name, "--version", version, "--force"
  ], check=True)
  print(f"Model packaged successfully in {output_dir}")
  parser = argparse.ArgumentParser(description="Standalone spaCy V3 Training Script")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  
  # Create config command
  config_parser = subparsers.add_parser("create-config", help="Create a default config file")
  config_parser.add_argument("--output", "-o", required=True, help="Output config path")
  config_parser.add_argument("--lang", default="tl", help="Language code")
  config_parser.add_argument("--pipeline", default="ner", help="Pipeline components")
  
  # Initialize model command
  init_parser = subparsers.add_parser("init", help="Initialize model from config")
  init_parser.add_argument("--config", "-c", required=True, help="Config file path")
  init_parser.add_argument("--output", "-o", required=True, help="Output directory")
  
  # Train command
  train_parser = subparsers.add_parser("train", help="Train a model")
  train_parser.add_argument("--config", "-c", required=True, help="Config file path")
  train_parser.add_argument("--output", "-o", required=True, help="Output directory")
  train_parser.add_argument("--train-path", help="Override training data path")
  train_parser.add_argument("--dev-path", help="Override dev data path")
  
  # Evaluate command
  eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
  eval_parser.add_argument("--model", "-m", required=True, help="Model path")
  eval_parser.add_argument("--data", "-d", required=True, help="Test data path")
  eval_parser.add_argument("--output", "-o", help="Output file for results")
  
  # Package command
  package_parser = subparsers.add_parser("package", help="Package a trained model")
  package_parser.add_argument("--input", "-i", required=True, help="Input model directory")
  package_parser.add_argument("--output", "-o", required=True, help="Output directory")
  package_parser.add_argument("--name", "-n", required=True, help="Package name")
  package_parser.add_argument("--version", "-v", default="0.0.1", help="Package version")
  
  args = parser.parse_args()
  
  if args.command == "create-config":
    create_config(args.output, args.lang, args.pipeline)
  
  elif args.command == "init":
    initialize_model(args.config, args.output)
  
  elif args.command == "train":
    overrides = {}
    if args.train_path:
      overrides["paths.train"] = args.train_path
    if args.dev_path:
      overrides["paths.dev"] = args.dev_path
    train_model(args.config, args.output, overrides)
  
  elif args.command == "evaluate":
    evaluate_model(args.model, args.data, args.output)
  
  elif args.command == "package":
    package_model(args.input, args.output, args.name, args.version)
  
  else:
    parser.print_help()
    sys.exit(1)


def main():
  parser = argparse.ArgumentParser(description="Standalone spaCy V3 Training Script")
  subparsers = parser.add_subparsers(dest="command", help="Available commands")
  
  # Create config command
  config_parser = subparsers.add_parser("create-config", help="Create a default config file")
  config_parser.add_argument("--output", "-o", required=True, help="Output config path")
  config_parser.add_argument("--lang", default="tl", help="Language code")
  config_parser.add_argument("--pipeline", default="ner", help="Pipeline components")
  
  # Initialize model command
  init_parser = subparsers.add_parser("init", help="Initialize model from config")
  init_parser.add_argument("--config", "-c", required=True, help="Config file path")
  init_parser.add_argument("--output", "-o", required=True, help="Output directory")
  
  # Train command
  train_parser = subparsers.add_parser("train", help="Train a model")
  train_parser.add_argument("--config", "-c", required=True, help="Config file path")
  train_parser.add_argument("--output", "-o", required=True, help="Output directory")
  train_parser.add_argument("--train-path", help="Override training data path")
  train_parser.add_argument("--dev-path", help="Override dev data path")
  
  # Evaluate command
  eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
  eval_parser.add_argument("--model", "-m", required=True, help="Model path")
  eval_parser.add_argument("--data", "-d", required=True, help="Test data path")
  eval_parser.add_argument("--output", "-o", help="Output file for results")
  
  # Package command
  package_parser = subparsers.add_parser("package", help="Package a trained model")
  package_parser.add_argument("--input", "-i", required=True, help="Input model directory")
  package_parser.add_argument("--output", "-o", required=True, help="Output directory")
  package_parser.add_argument("--name", "-n", required=True, help="Package name")
  package_parser.add_argument("--version", "-v", default="0.0.1", help="Package version")
  
  args = parser.parse_args()
  
  if args.command == "create-config":
    create_config(args.output, args.lang, args.pipeline)
  
  elif args.command == "init":
    initialize_model(args.config, args.output)
  
  elif args.command == "train":
    overrides = {}
    if args.train_path:
      overrides["paths.train"] = args.train_path
    if args.dev_path:
      overrides["paths.dev"] = args.dev_path
    train_model(args.config, args.output, overrides)
  
  elif args.command == "evaluate":
    evaluate_model(args.model, args.data, args.output)
  
  elif args.command == "package":
    package_model(args.input, args.output, args.name, args.version)
  
  else:
    parser.print_help()
    sys.exit(1)

if __name__ == "__main__":
  main()