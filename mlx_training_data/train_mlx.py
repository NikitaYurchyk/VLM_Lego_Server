#!/usr/bin/env python3
"""
MLX LoRA Fine-tuning Script for Lego Assembly Instructions
"""

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.utils import load, apply_repetition_penalty
import json
import argparse
from pathlib import Path

def train_lora(config_path: str):
    """Train LoRA adapter using MLX."""
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Starting MLX LoRA training...")
    print(f"Model: {config['model']}")
    print(f"Data: {config['data']}")
    print(f"LoRA layers: {config['lora_layers']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # MLX training command
    import subprocess
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", config["model"],
        "--data", config["data"],
        "--lora-layers", str(config["lora_layers"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--num-epochs", str(config["num_epochs"]),
        "--steps-per-eval", str(config["steps_per_eval"]),
        "--save-every", str(config["save_every"]),
        "--adapter-path", config["adapter_path"],
        "--max-seq-len", str(config["max_seq_len"])
    ]
    
    if "valid_data" in config:
        cmd.extend(["--valid-data", config["valid_data"]])
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Training completed successfully!")
        print("Adapter saved to:", config["adapter_path"])
    else:
        print("Training failed:")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
    return result.returncode == 0

def test_model(adapter_path: str, base_model: str, test_prompt: str):
    """Test the fine-tuned model."""
    
    print(f"Loading model with adapter: {adapter_path}")
    
    # Load model and adapter
    model, tokenizer = load(base_model, adapter_path=adapter_path)
    
    # Generate response
    print(f"Test prompt: {test_prompt}")
    print("Generating response...")
    
    response = generate(
        model, tokenizer, 
        prompt=test_prompt,
        max_tokens=200,
        temp=0.7
    )
    
    print(f"Response: {response}")

def main():
    parser = argparse.ArgumentParser(description="MLX LoRA Training for Lego Instructions")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--adapter-path", help="Path to trained adapter (for testing)")
    parser.add_argument("--prompt", default="Please provide assembly instructions for the next Lego step.", help="Test prompt")
    
    args = parser.parse_args()
    
    if args.test:
        if not args.adapter_path:
            print("Error: --adapter-path required for testing")
            return
        
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        test_model(args.adapter_path, config["model"], args.prompt)
    else:
        success = train_lora(args.config)
        if not success:
            print("Training failed!")
            return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
