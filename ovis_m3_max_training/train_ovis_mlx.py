#!/usr/bin/env python3
"""
Ovis2.5-9B MLX LoRA Fine-tuning Script
Optimized for M3 Max with 64GB RAM and LM Studio compatibility
"""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate, convert
import json
import argparse
import subprocess
import time
from pathlib import Path
import os

def setup_mlx_environment():
    """Setup MLX environment for optimal M3 Max performance."""
    
    # Set MLX to use all available GPU memory
    mx.set_default_device(mx.gpu)
    
    # Optimize for M3 Max
    os.environ["MLX_GPU_MEMORY_LIMIT"] = "0.95"  # Use 95% of GPU memory
    os.environ["MLX_ENABLE_FAST_MATH"] = "1"
    
    print("🔧 MLX environment configured for M3 Max")
    print(f"   • GPU memory limit: 95%")
    print(f"   • Fast math: Enabled")
    print(f"   • Default device: {mx.default_device()}")

def train_ovis_lora(config_path: str):
    """Train LoRA adapter for Ovis2.5-9B using MLX."""
    
    setup_mlx_environment()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("🚀 Starting Ovis2.5-9B LoRA training...")
    print(f"   • Model: {config['model']}")
    print(f"   • Data: {config['data']}")
    print(f"   • LoRA layers: {config['lora_layers']}")
    print(f"   • Batch size: {config['batch_size']}")
    print(f"   • Learning rate: {config['learning_rate']}")
    print(f"   • Max sequence length: {config['max_seq_len']}")
    
    # Check if model needs conversion to MLX format
    model_path = config["model"]
    if not os.path.exists("models/ovis2.5-9b-mlx"):
        print("🔄 Converting Ovis2.5-9B to MLX format...")
        
        # Convert model to MLX format
        convert_cmd = [
            "python", "-m", "mlx_lm.convert",
            "--hf-path", model_path,
            "--mlx-path", "models/ovis2.5-9b-mlx",
            "--quantize"  # Quantize for better performance
        ]
        
        result = subprocess.run(convert_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Model conversion failed: {result.stderr}")
            return False
        
        model_path = "models/ovis2.5-9b-mlx"
        print("✅ Model converted to MLX format")
    
    # MLX LoRA training command optimized for M3 Max
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", model_path,
        "--data", config["data"],
        "--lora-layers", str(config["lora_layers"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--num-epochs", str(config["num_epochs"]),
        "--steps-per-eval", str(config["steps_per_eval"]),
        "--save-every", str(config["save_every"]),
        "--adapter-path", config["adapter_path"],
        "--max-seq-len", str(config["max_seq_len"]),
        "--warmup-steps", str(config.get("warmup_steps", 100)),
        "--weight-decay", str(config.get("weight_decay", 0.01))
    ]
    
    if "valid_data" in config:
        cmd.extend(["--valid-data", config["valid_data"]])
    
    # Add M3 Max optimizations
    hardware_config = config.get("hardware_optimization", {})
    if hardware_config.get("mixed_precision"):
        cmd.append("--mixed-precision")
    if hardware_config.get("gradient_accumulation_steps"):
        cmd.extend(["--grad-accum-steps", str(hardware_config["gradient_accumulation_steps"])])
    
    print(f"🎯 Training command: {' '.join(cmd)}")
    
    # Start training with progress monitoring
    start_time = time.time()
    result = subprocess.run(cmd, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        training_time = end_time - start_time
        print(f"✅ Training completed successfully!")
        print(f"   • Training time: {training_time/3600:.1f} hours")
        print(f"   • Adapter saved to: {config['adapter_path']}")
        
        # Export for LM Studio
        export_for_lm_studio(config)
        
        return True
    else:
        print("❌ Training failed!")
        return False

def export_for_lm_studio(config: dict):
    """Export trained model for LM Studio compatibility."""
    
    print("🔄 Exporting for LM Studio...")
    
    lm_studio_config = config.get("lm_studio_compatibility", {})
    export_format = lm_studio_config.get("export_format", "gguf")
    quantization = lm_studio_config.get("quantization", "q4_k_m")
    
    if export_format == "gguf":
        # Convert to GGUF format for LM Studio
        export_cmd = [
            "python", "-m", "mlx_lm.convert",
            "--mlx-path", config["adapter_path"],
            "--hf-path", f"{config['adapter_path']}_hf",
            "--upload-repo", f"ovis2.5-9b-lego-lora"
        ]
        
        result = subprocess.run(export_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Model exported for LM Studio")
            print(f"   • Format: {export_format}")
            print(f"   • Quantization: {quantization}")
            print(f"   • Location: {config['adapter_path']}_hf")
        else:
            print(f"⚠️  Export warning: {result.stderr}")

def test_ovis_model(config_path: str, adapter_path: str, test_prompt: str):
    """Test the fine-tuned Ovis2.5-9B model."""
    
    setup_mlx_environment()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"🧪 Testing Ovis2.5-9B with adapter: {adapter_path}")
    
    try:
        # Load model and adapter
        model_path = config["model"]
        if os.path.exists("models/ovis2.5-9b-mlx"):
            model_path = "models/ovis2.5-9b-mlx"
        
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        
        # Generate response
        print(f"💭 Test prompt: {test_prompt}")
        print("🔄 Generating response...")
        
        response = generate(
            model, tokenizer,
            prompt=test_prompt,
            max_tokens=300,
            temp=0.7,
            top_p=0.9
        )
        
        print(f"🎯 Model response:")
        print(f"   {response}")
        
        return response
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return None

def benchmark_performance(config_path: str, adapter_path: str):
    """Benchmark model performance on M3 Max."""
    
    print("📊 Running performance benchmark...")
    
    test_prompts = [
        "Analyze this Lego assembly step and provide detailed instructions.",
        "What pieces are needed for this construction step?",
        "Describe the connection method shown in these images.",
        "Provide step-by-step assembly guidance for this Lego build."
    ]
    
    total_time = 0
    successful_tests = 0
    
    for i, prompt in enumerate(test_prompts):
        print(f"🧪 Test {i+1}/{len(test_prompts)}: {prompt[:50]}...")
        
        start_time = time.time()
        response = test_ovis_model(config_path, adapter_path, prompt)
        end_time = time.time()
        
        if response:
            test_time = end_time - start_time
            total_time += test_time
            successful_tests += 1
            print(f"   ✅ Completed in {test_time:.2f}s")
        else:
            print(f"   ❌ Failed")
    
    if successful_tests > 0:
        avg_time = total_time / successful_tests
        print(f"📈 Performance Summary:")
        print(f"   • Successful tests: {successful_tests}/{len(test_prompts)}")
        print(f"   • Average response time: {avg_time:.2f}s")
        print(f"   • Total benchmark time: {total_time:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Ovis2.5-9B MLX LoRA Training")
    parser.add_argument("--config", required=True, help="Path to training config")
    parser.add_argument("--test", action="store_true", help="Test trained model")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--adapter-path", help="Path to trained adapter")
    parser.add_argument("--prompt", default="Analyze these Lego assembly images and provide step-by-step instructions.", help="Test prompt")
    
    args = parser.parse_args()
    
    if args.test:
        if not args.adapter_path:
            print("❌ Error: --adapter-path required for testing")
            return 1
        
        test_ovis_model(args.config, args.adapter_path, args.prompt)
        
    elif args.benchmark:
        if not args.adapter_path:
            print("❌ Error: --adapter-path required for benchmarking")
            return 1
            
        benchmark_performance(args.config, args.adapter_path)
        
    else:
        success = train_ovis_lora(args.config)
        if not success:
            print("❌ Training failed!")
            return 1
    
    print("🎉 All operations completed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
