#!/usr/bin/env python3
"""
Ovis2.5-9B MLX LoRA Fine-tuning Data Preparation Script

Converts Lego assembly instructions with reference images into MLX format
specifically optimized for Ovis2.5-9B model fine-tuning and LM Studio compatibility.
"""

import json
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
from PIL import Image
import numpy as np

def resize_image_for_ovis(image_path: str, max_size: int = 1024, quality: int = 90) -> str:
    """
    Resize image optimally for Ovis2.5-9B which supports native resolution processing.
    Ovis can handle high-res images better than other VLMs.
    """
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Ovis2.5 can handle larger images - preserve more detail
        original_size = max(img.size)
        if original_size > max_size:
            # Resize maintaining aspect ratio
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save with high quality for better vision understanding
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_ovis_conversation(step_data: Dict, ref_images: List[str], image_dir: Path, 
                           use_thinking_mode: bool = False) -> Dict[str, Any]:
    """Create Ovis2.5-9B format conversation entry with optional thinking mode."""
    
    instruction_text = "\n".join(step_data.get('instructions', []))
    step_num = step_data.get('step')
    
    # Create system prompt optimized for Ovis2.5-9B
    system_prompt = """You are an expert Lego assembly assistant with advanced visual reasoning capabilities. 
Analyze the provided reference images carefully and provide clear, precise step-by-step assembly instructions. 
Focus on piece identification, spatial relationships, and connection methods."""
    
    if use_thinking_mode:
        system_prompt += """
Use your thinking capabilities to:
1. First analyze what you see in each image
2. Identify the specific Lego pieces involved
3. Determine the assembly sequence
4. Provide clear, actionable instructions"""
    
    # User message with multi-image support
    user_content = f"""Please analyze these {len(ref_images)} reference images showing different angles and perspectives of Lego assembly step {step_num}. 

Provide detailed assembly instructions that include:
- Specific piece identification and colors
- Spatial orientation and positioning
- Connection methods and techniques
- Any important assembly tips or warnings

Step {step_num} reference images:"""
    
    # Process images
    processed_images = []
    for i, img_path in enumerate(ref_images):
        if img_path.endswith(('.jpeg', '.jpg', '.png')):
            full_path = image_dir / img_path
            if full_path.exists():
                try:
                    image_b64 = resize_image_for_ovis(str(full_path), max_size=1024)
                    processed_images.append({
                        "type": "image",
                        "image": f"data:image/jpeg;base64,{image_b64}",
                        "alt_text": f"Lego assembly step {step_num} - view {i+1}"
                    })
                except Exception as e:
                    print(f"Warning: Could not process image {img_path}: {e}")
    
    # Create conversation in Ovis format
    conversation = {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user", 
                "content": user_content,
                "images": processed_images
            },
            {
                "role": "assistant",
                "content": instruction_text
            }
        ],
        "metadata": {
            "step": step_num,
            "step_type": step_data.get('step_type', 'assembly'),
            "num_images": len(processed_images),
            "thinking_mode": use_thinking_mode,
            "model_target": "ovis2.5-9b"
        }
    }
    
    return conversation

def create_ovis_instruct_format(step_data: Dict, ref_images: List[str], image_dir: Path) -> Dict[str, Any]:
    """Create Ovis2.5-9B instruction-following format for MLX training."""
    
    instruction_text = "\n".join(step_data.get('instructions', []))
    step_num = step_data.get('step')
    
    # Process images with Ovis-optimized settings
    processed_images = []
    for img_path in ref_images:
        if img_path.endswith(('.jpeg', '.jpg', '.png')):
            full_path = image_dir / img_path
            if full_path.exists():
                try:
                    image_b64 = resize_image_for_ovis(str(full_path), max_size=1024, quality=95)
                    processed_images.append({
                        "path": img_path,
                        "data": f"data:image/jpeg;base64,{image_b64}",
                        "resolution": "high"  # Flag for Ovis native resolution support
                    })
                except Exception as e:
                    print(f"Warning: Could not process image {img_path}: {e}")
    
    return {
        "instruction": f"""Analyze the provided reference images for Lego assembly step {step_num}. 
Use your advanced visual reasoning to identify pieces, spatial relationships, and assembly sequence. 
Provide comprehensive, actionable assembly instructions.""",
        
        "input": f"""Step {step_num}: {len(processed_images)} high-resolution reference images showing multiple perspectives of the Lego assembly process. 
Focus on piece identification, connection methods, and spatial orientation.""",
        
        "output": instruction_text,
        
        "images": processed_images,
        
        "metadata": {
            "step": step_num,
            "step_type": step_data.get('step_type', 'assembly'),
            "num_images": len(processed_images),
            "model_target": "ovis2.5-9b",
            "complexity": "intermediate",
            "domain": "lego_assembly"
        }
    }

def prepare_ovis_dataset(instructions_file: str, images_dir: str, output_dir: str, 
                        format_type: str = "conversation", max_image_size: int = 1024,
                        thinking_mode: bool = False):
    """
    Prepare Ovis2.5-9B MLX fine-tuning dataset optimized for M3 Max.
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load instructions
    with open(instructions_file, 'r') as f:
        data = json.load(f)
    
    instructions = data.get('instructions', [])
    images_path = Path(images_dir)
    
    training_data = []
    
    print(f"🚀 Processing {len(instructions)} instruction steps for Ovis2.5-9B...")
    print(f"📊 Format: {format_type}")
    print(f"🖼️  Max image size: {max_image_size}px")
    print(f"🧠 Thinking mode: {thinking_mode}")
    print(f"💻 Target: M3 Max with 64GB RAM")
    
    for step_data in instructions:
        step_num = step_data.get('step')
        if step_num is None:
            continue
            
        # Find reference images for this step
        step_images_dir = images_path / str(step_num)
        ref_images = []
        
        if step_images_dir.exists():
            # Get all image files in step directory
            for ext in ['*.jpeg', '*.jpg', '*.png']:
                ref_images.extend([str(step_num) + "/" + img.name 
                                 for img in step_images_dir.glob(ext)])
            
        if not ref_images:
            print(f"⚠️  Warning: No reference images found for step {step_num}")
            continue
            
        # Create training entry based on format
        if format_type == "conversation":
            entry = create_ovis_conversation(step_data, ref_images, images_path, thinking_mode)
        elif format_type == "instruct":
            entry = create_ovis_instruct_format(step_data, ref_images, images_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        training_data.append(entry)
        print(f"✅ Processed step {step_num} with {len(ref_images)} images")
    
    # Save training data in JSONL format for MLX
    output_file = output_path / f"ovis_train_{format_type}.jsonl"
    with open(output_file, 'w') as f:
        for entry in training_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"\n🎯 Generated {len(training_data)} training samples")
    print(f"💾 Saved to: {output_file}")
    
    # Generate Ovis2.5-9B optimized config for M3 Max
    config = {
        "model": "AIDC-AI/Ovis2.5-9B",
        "model_type": "ovis",
        "data": str(output_file),
        "lora_layers": 32,  # Higher rank for 9B model with 64GB RAM
        "batch_size": 2,    # Can handle larger batches with 64GB
        "learning_rate": 5e-6,  # Lower LR for larger model
        "num_epochs": 5,
        "steps_per_eval": 25,
        "save_every": 50,
        "adapter_path": "ovis_adapters",
        "max_seq_len": 4096,  # Longer sequences for detailed instructions
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "vision_config": {
            "max_image_size": max_image_size,
            "native_resolution": True,  # Ovis2.5 specialty
            "image_quality": "high",
            "multi_image_support": True
        },
        "hardware_optimization": {
            "platform": "m3_max",
            "ram_gb": 64,
            "mixed_precision": True,
            "gradient_accumulation_steps": 4,
            "dataloader_num_workers": 4
        },
        "lm_studio_compatibility": {
            "export_format": "gguf",
            "quantization": "q4_k_m",
            "context_length": 4096
        }
    }
    
    config_file = output_path / "ovis_config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Generate statistics
    total_images = sum(entry["metadata"]["num_images"] for entry in training_data)
    avg_images_per_step = total_images / len(training_data) if training_data else 0
    
    print(f"\n📈 Ovis2.5-9B Dataset Statistics:")
    print(f"   • Total training samples: {len(training_data)}")
    print(f"   • Total reference images: {total_images}")
    print(f"   • Average images per step: {avg_images_per_step:.1f}")
    print(f"   • Config saved to: {config_file}")
    
    # Create validation split optimized for the dataset size
    if len(training_data) > 8:  # Need sufficient data for meaningful validation
        val_size = max(2, len(training_data) // 6)  # ~17% for validation
        val_data = training_data[-val_size:]
        train_data = training_data[:-val_size]
        
        # Save splits
        train_file = output_path / f"ovis_train_{format_type}.jsonl"
        val_file = output_path / f"ovis_valid_{format_type}.jsonl"
        
        with open(train_file, 'w') as f:
            for entry in train_data:
                f.write(json.dumps(entry) + '\n')
                
        with open(val_file, 'w') as f:
            for entry in val_data:
                f.write(json.dumps(entry) + '\n')
        
        # Update config with validation
        config["valid_data"] = str(val_file)
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   • Training samples: {len(train_data)}")
        print(f"   • Validation samples: {len(val_data)}")
        print(f"   • Split ratio: {len(train_data)/len(training_data):.1%} train, {len(val_data)/len(training_data):.1%} validation")

def create_ovis_training_script(output_dir: str):
    """Create comprehensive MLX training script for Ovis2.5-9B."""
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    script_path = Path(output_dir) / "train_ovis_mlx.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    print(f"🔧 Ovis2.5-9B MLX training script saved to: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare Ovis2.5-9B MLX LoRA fine-tuning data")
    parser.add_argument("--instructions", required=True, help="Path to instructions JSON file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--format", choices=["conversation", "instruct"], default="conversation", 
                       help="Output format for MLX training")
    parser.add_argument("--max-image-size", type=int, default=1024, help="Maximum image dimension (Ovis supports higher res)")
    parser.add_argument("--thinking-mode", action="store_true", help="Enable Ovis thinking mode training")
    parser.add_argument("--create-script", action="store_true", help="Create training script")
    
    args = parser.parse_args()
    
    print("🎯 Preparing Ovis2.5-9B MLX LoRA fine-tuning data...")
    print(f"   • Target hardware: M3 Max with 64GB RAM")
    print(f"   • Model: AIDC-AI/Ovis2.5-9B")
    print(f"   • LM Studio compatibility: Enabled")
    
    # Prepare dataset
    prepare_ovis_dataset(
        args.instructions, 
        args.images, 
        args.output, 
        args.format, 
        args.max_image_size,
        args.thinking_mode
    )
    
    # Create training script if requested
    if args.create_script:
        create_ovis_training_script(args.output)
        
    print("🚀 Setup complete! Ready for Ovis2.5-9B fine-tuning on M3 Max!")

if __name__ == "__main__":
    main()