# MLX LoRA Fine-tuning Guide for Lego Assembly Instructions

This guide shows how to fine-tune a vision-language model using MLX on Apple Silicon for Lego assembly instruction generation.

## Quick Start

1. **Install MLX dependencies:**
```bash
pip install mlx-lm pillow
```

2. **Prepare training data:**
```bash
python3 prepare_mlx_data.py \
  --instructions instructions_with_more_ref_imgs/lego-60399-green-race-car-readscr.json \
  --images instructions_with_more_ref_imgs/images \
  --output mlx_training_data \
  --format conversation \
  --create-script
```

3. **Start training:**
```bash
cd mlx_training_data
python3 train_mlx.py --config config.json
```

## Data Formats

### Conversation Format
Best for chat-based interactions:
```jsonl
{
  "messages": [
    {"role": "user", "content": "Please analyze these reference images and provide step-by-step assembly instructions for Lego step 12."},
    {"role": "assistant", "content": "Assembly instructions here..."}
  ],
  "images": [
    {"path": "12/photo1.jpeg", "data": "data:image/jpeg;base64,..."}
  ],
  "metadata": {"step": 12, "num_images": 5}
}
```

### Instruction Format  
Traditional instruction-following format:
```jsonl
{
  "instruction": "Analyze the reference images and provide clear assembly instructions for Lego construction step 12.",
  "input": "Step 12 reference images showing different angles...",
  "output": "Assembly instructions here...",
  "images": [{"path": "12/photo1.jpeg", "data": "data:image/jpeg;base64,..."}],
  "metadata": {"step": 12, "num_images": 5}
}
```

## Training Configuration

Your generated `config.json`:
```json
{
  "model": "mlx-community/llava-1.5-7b-4bit",
  "data": "train_conversation.jsonl",
  "valid_data": "valid_conversation.jsonl", 
  "lora_layers": 16,
  "batch_size": 1,
  "learning_rate": 1e-5,
  "num_epochs": 3,
  "steps_per_eval": 50,
  "save_every": 100,
  "adapter_path": "adapters",
  "max_seq_len": 2048,
  "vision_config": {
    "max_image_size": 512,
    "image_token": "<image>"
  }
}
```

## Training Commands

### Basic Training
```bash
cd mlx_training_data
python3 train_mlx.py --config config.json
```

### Training with Custom Parameters
```bash
python3 -m mlx_lm.lora \
  --model mlx-community/llava-1.5-7b-4bit \
  --data train_conversation.jsonl \
  --valid-data valid_conversation.jsonl \
  --lora-layers 16 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --num-epochs 3 \
  --steps-per-eval 50 \
  --save-every 100 \
  --adapter-path adapters \
  --max-seq-len 2048
```

### Resume Training
```bash
python3 -m mlx_lm.lora \
  --model mlx-community/llava-1.5-7b-4bit \
  --data train_conversation.jsonl \
  --resume-adapter-file adapters/adapters.npz
```

## Testing the Model

### Using the Generated Script
```bash
cd mlx_training_data
python3 train_mlx.py \
  --config config.json \
  --test \
  --adapter-path adapters \
  --prompt "Please provide assembly instructions for the next Lego step."
```

### Using MLX-LM Directly
```bash
python3 -m mlx_lm.generate \
  --model mlx-community/llava-1.5-7b-4bit \
  --adapter-path mlx_training_data/adapters \
  --prompt "Please provide assembly instructions for Lego step 15." \
  --max-tokens 200 \
  --temp 0.7
```

### Python Testing Script
```python
from mlx_lm import load, generate

# Load model with trained adapter
model, tokenizer = load(
    "mlx-community/llava-1.5-7b-4bit",
    adapter_path="mlx_training_data/adapters"
)

# Generate response
prompt = "Please provide assembly instructions for the next Lego step."
response = generate(
    model, tokenizer,
    prompt=prompt,
    max_tokens=200,
    temp=0.7
)

print(f"Model response: {response}")
```

## Performance Optimization

### Memory Optimization
- Use 4-bit quantized models: `mlx-community/llava-1.5-7b-4bit`
- Reduce batch size if running out of memory: `--batch-size 1`
- Lower image resolution: `--max-image-size 512`

### Training Speed
- Increase batch size if you have enough memory: `--batch-size 2`
- Use gradient accumulation for effective larger batches
- Reduce `--steps-per-eval` for faster feedback

### Model Quality
- Increase LoRA rank: `--lora-layers 32` (uses more memory)
- Train for more epochs: `--num-epochs 5`
- Use higher resolution images: `--max-image-size 768`

## Dataset Statistics

Your current dataset:
- **21 training samples** (17 train, 4 validation)
- **111 reference images** total  
- **5.3 images per step** average
- **Steps covered:** 12, 13, 14, 15, 16, 17, 18

## Model Performance Monitoring

### Training Metrics
Monitor these during training:
- **Loss**: Should decrease over time
- **Perplexity**: Lower is better
- **Validation loss**: Watch for overfitting

### Evaluation
```bash
# Generate sample outputs during training
python3 -m mlx_lm.generate \
  --model mlx-community/llava-1.5-7b-4bit \
  --adapter-path adapters \
  --prompt "Analyze this Lego assembly step:" \
  --max-tokens 100
```

## Common Issues & Solutions

### Out of Memory
```bash
# Reduce batch size
--batch-size 1

# Use smaller images  
--max-image-size 256

# Use gradient checkpointing
--grad-checkpoint
```

### Poor Quality Outputs
```bash
# Increase LoRA layers
--lora-layers 32

# Train longer
--num-epochs 5

# Lower learning rate
--learning-rate 5e-6
```

### Slow Training
```bash
# Increase batch size (if memory allows)
--batch-size 2

# Reduce validation frequency
--steps-per-eval 100

# Use mixed precision
--mixed-precision
```

## Directory Structure After Training

```
mlx_training_data/
├── config.json              # Training configuration
├── train_conversation.jsonl # Training data
├── valid_conversation.jsonl # Validation data  
├── train_mlx.py             # Training script
└── adapters/                # Trained LoRA adapters
    ├── adapters.npz         # Main adapter weights
    ├── config.json          # Adapter config
    └── tokenizer_config.json# Tokenizer config
```

## Integration with Your App

### Loading in Python
```python
from mlx_lm import load, generate

class LegoAssemblyAssistant:
    def __init__(self, adapter_path):
        self.model, self.tokenizer = load(
            "mlx-community/llava-1.5-7b-4bit",
            adapter_path=adapter_path
        )
    
    def get_instructions(self, step_number, context=""):
        prompt = f"Please provide assembly instructions for Lego step {step_number}. {context}"
        
        response = generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=200,
            temp=0.7
        )
        
        return response

# Usage
assistant = LegoAssemblyAssistant("mlx_training_data/adapters")
instructions = assistant.get_instructions(15, "Focus on piece connections.")
```

### FastAPI Integration
```python
from fastapi import FastAPI
from mlx_lm import load, generate

app = FastAPI()
model, tokenizer = load("mlx-community/llava-1.5-7b-4bit", adapter_path="mlx_training_data/adapters")

@app.post("/generate_instructions")
async def generate_instructions(step: int, context: str = ""):
    prompt = f"Please provide assembly instructions for Lego step {step}. {context}"
    
    response = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=200,
        temp=0.7
    )
    
    return {"instructions": response, "step": step}
```

## Next Steps

1. **Expand Dataset**: Add more Lego sets and instruction types
2. **Data Augmentation**: Create variations of existing instructions
3. **Multi-Modal**: Include 3D model information
4. **Evaluation**: Create test sets for systematic evaluation
5. **Deployment**: Integrate with your AR app via FastAPI

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX-LM Repository](https://github.com/ml-explore/mlx-examples/tree/main/llms)
- [LLaVA Model Details](https://huggingface.co/llava-hf/llava-1.5-7b-hf)