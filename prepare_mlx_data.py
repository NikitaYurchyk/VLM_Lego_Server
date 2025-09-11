#!/usr/bin/env python3
"""
MLX LoRA Fine-tuning Data Preparation Script

Converts Lego assembly instructions with reference images into MLX format
suitable for vision-language model fine-tuning.
"""

import json
import os
import base64
from pathlib import Path
from typing import List, Dict, Any
import argparse
from PIL import Image
import numpy as np

def resize_image(image_path: str, max_size: int = 512) -> str:
    """Resize image and convert to base64 for MLX training."""
    with Image.open(image_path) as img:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize maintaining aspect ratio
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Save to bytes and encode
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

def create_mlx_conversation(step_data: Dict, ref_images: List[str], image_dir: Path) -> Dict[str, Any]:
    """Create MLX-format conversation entry."""
    
    # Prepare instruction text
    instruction_text = "\n".join(step_data.get('instructions', []))
    step_num = step_data.get('step')
    
    # Create conversation in MLX format
    conversation = {
        "messages": [
            {
                "role": "user",
                "content": f"Please analyze these reference images and provide step-by-step assembly instructions for Lego step {step_num}."
            },
            {
                "role": "assistant", 
                "content": instruction_text
            }
        ],
        "images": [],
        "metadata": {
            "step": step_num,
            "step_type": step_data.get('step_type', 'assembly'),
            "num_images": 0
        }
    }
    
    # Add images
    for img_path in ref_images:
        if img_path.endswith(('.jpeg', '.jpg', '.png')):
            full_path = image_dir / img_path
            if full_path.exists():
                try:
                    # Resize and encode image for MLX
                    image_b64 = resize_image(str(full_path))
                    conversation["images"].append({
                        "path": img_path,
                        "data": f"data:image/jpeg;base64,{image_b64}"
                    })
                except Exception as e:
                    print(f"Warning: Could not process image {img_path}: {e}")
    
    conversation["metadata"]["num_images"] = len(conversation["images"])
    return conversation

def create_mlx_instruct_format(step_data: Dict, ref_images: List[str], image_dir: Path) -> Dict[str, Any]:
    """Create MLX instruction-following format."""
    
    instruction_text = "\n".join(step_data.get('instructions', []))
    step_num = step_data.get('step')
    
    # Process images
    processed_images = []
    for img_path in ref_images:
        if img_path.endswith(('.jpeg', '.jpg', '.png')):
            full_path = image_dir / img_path
            if full_path.exists():
                try:
                    image_b64 = resize_image(str(full_path))
                    processed_images.append({
                        "path": img_path,
                        "data": f"data:image/jpeg;base64,{image_b64}"
                    })
                except Exception as e:
                    print(f"Warning: Could not process image {img_path}: {e}")
    
    return {
        "instruction": f"Analyze the reference images and provide clear assembly instructions for Lego construction step {step_num}. Be specific about piece placement and connections.",
        "input": f"Step {step_num} reference images showing different angles and perspectives of the assembly process.",
        "output": instruction_text,
        "images": processed_images,
        "metadata": {
            "step": step_num,
            "step_type": step_data.get('step_type', 'assembly'),
            "num_images": len(processed_images)
        }
    }

def prepare_mlx_dataset(instructions_file: str, images_dir: str, output_dir: str, 
                       format_type: str = "conversation", max_image_size: int = 512):
    """
    Prepare MLX fine-tuning dataset from Lego instructions.
    
    Args:
        instructions_file: Path to JSON instructions file
        images_dir: Path to directory containing reference images
        output_dir: Output directory for MLX training data
        format_type: Format type ("conversation" or "instruct")
        max_image_size: Maximum image dimension
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
    
    print(f"Processing {len(instructions)} instruction steps for MLX...")
    print(f"Using format: {format_type}")
    print(f"Max image size: {max_image_size}px")
    
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
            print(f"Warning: No reference images found for step {step_num}")
            continue
            
        # Create training entry based on format
        if format_type == "conversation":
            entry = create_mlx_conversation(step_data, ref_images, images_path)
        elif format_type == "instruct":
            entry = create_mlx_instruct_format(step_data, ref_images, images_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
            
        training_data.append(entry)
    
    # Save training data
    output_file = output_path / f"train_{format_type}.jsonl"
    with open(output_file, 'w') as f:
        for entry in training_data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Generated {len(training_data)} training samples")
    print(f"Saved to: {output_file}")
    
    # Generate statistics and config
    total_images = sum(entry["metadata"]["num_images"] for entry in training_data)
    avg_images_per_step = total_images / len(training_data) if training_data else 0
    
    # Create MLX config
    config = {
        "model": "mlx-community/llava-1.5-7b-4bit",
        "data": str(output_file),
        "lora_layers": 16,
        "batch_size": 1,
        "learning_rate": 1e-5,
        "num_epochs": 3,
        "steps_per_eval": 50,
        "save_every": 100,
        "adapter_path": "adapters",
        "max_seq_len": 2048,
        "vision_config": {
            "max_image_size": max_image_size,
            "image_token": "<image>"
        }
    }
    
    config_file = output_path / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nMLX Dataset Statistics:")
    print(f"- Total training samples: {len(training_data)}")
    print(f"- Total reference images: {total_images}")
    print(f"- Average images per step: {avg_images_per_step:.1f}")
    print(f"- Config saved to: {config_file}")
    
    # Create validation split
    if len(training_data) > 5:
        val_size = max(1, len(training_data) // 5)  # 20% for validation
        val_data = training_data[-val_size:]
        train_data = training_data[:-val_size]
        
        # Save splits
        train_file = output_path / f"train_{format_type}.jsonl"
        val_file = output_path / f"valid_{format_type}.jsonl"
        
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
        
        print(f"- Training samples: {len(train_data)}")
        print(f"- Validation samples: {len(val_data)}")

def create_training_script(output_dir: str):
    """Create MLX training script."""
    
    script_content = '''#!/usr/bin/env python3
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
'''
    
    script_path = Path(output_dir) / "train_mlx.py"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    script_path.chmod(0o755)
    print(f"MLX training script saved to: {script_path}")

def main():
    parser = argparse.ArgumentParser(description="Prepare MLX LoRA fine-tuning data from Lego instructions")
    parser.add_argument("--instructions", required=True, help="Path to instructions JSON file")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Output directory path")
    parser.add_argument("--format", choices=["conversation", "instruct"], default="conversation", 
                       help="Output format for MLX training")
    parser.add_argument("--max-image-size", type=int, default=512, help="Maximum image dimension")
    parser.add_argument("--create-script", action="store_true", help="Create training script")
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_mlx_dataset(args.instructions, args.images, args.output, args.format, args.max_image_size)
    
    # Create training script if requested
    if args.create_script:
        create_training_script(args.output)

if __name__ == "__main__":
    main()