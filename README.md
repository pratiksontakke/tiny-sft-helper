# Tiny Supervised Fine-Tuning (SFT) Project

This project demonstrates fine-tuning a small language model to be a polite and ethical AI assistant.

## Model

- Base Model: `NousResearch/Meta-Llama-3-8B`
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Training Epochs: 5
- Learning Rate: 1e-4

## Dataset

The training dataset (`dataset.jsonl`) contains 20 carefully curated examples that include:

1. Factual Q&A (e.g., "What is the capital of France?")
2. Polite-tone responses (e.g., "Could you please explain...")
3. Short vs. Long-form answers (demonstrating length control)
4. Ethical refusal cases (e.g., declining to assist with hacking)

## Training Configuration

The LoRA configuration includes:
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target Modules: 
  - Query, Key, Value, and Output projections
  - Gate, Up, and Down projections

Additional training improvements:
- Weight decay: 0.01
- Gradient clipping: 1.0
- Increased warmup ratio: 0.1
- Cosine learning rate schedule
- Increased context length: 512 tokens

## Files

- `dataset.jsonl`: Training dataset with prompt-response pairs
- `train.py`: Training script with LoRA configuration
- `evaluate.py`: Evaluation script for model comparison
- `before_after.md`: Side-by-side comparison of base and fine-tuned models
- `output/`: Directory containing model checkpoints and final weights

## Setup and Training

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
python train.py
```

3. Evaluate models:
```bash
python evaluate.py
```

## Results

The evaluation compares the base and fine-tuned models on:
1. Factual accuracy
2. Response politeness
3. Length control
4. Ethical behavior

See `before_after.md` for detailed comparison results.
