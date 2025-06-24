import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import os
import json

# Constants
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Changed to smaller model
OUTPUT_DIR = "output"
LORA_R = 8  # Reduced for smaller model
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
BATCH_SIZE = 1
MAX_LENGTH = 512
WEIGHT_DECAY = 0.01

def validate_jsonl(file_path):
    """Validate JSONL file format"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                json.loads(line.strip())
        return True
    except Exception as e:
        print(f"Error validating JSONL: {str(e)}")
        return False

def load_and_process_data():
    # Validate JSONL file first
    if not validate_jsonl('dataset.jsonl'):
        raise ValueError("Invalid JSONL file. Please fix the formatting issues.")
    
    # Load the dataset
    try:
        print("Loading dataset...")
        dataset = load_dataset('json', data_files='dataset.jsonl')
        print(f"Dataset loaded successfully. Number of examples: {len(dataset['train'])}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise
    
    # Initialize tokenizer
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
        padding_side="right",
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        if isinstance(examples, dict) and 'prompt' in examples and 'response' in examples:
            print("Processing batch of size:", len(examples['prompt']))
        
        # Combine prompt and response
        texts = []
        for i in range(len(examples['prompt'])):
            combined_text = examples['prompt'][i] + examples['response'][i]
            texts.append(combined_text)
        
        # Tokenize with increased max length
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Create labels (important for training)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=2,
        remove_columns=dataset["train"].column_names,
        num_proc=1
    )
    print("Dataset tokenization complete!")
    
    return tokenized_dataset, tokenizer

def setup_model_and_trainer(tokenizer, tokenized_dataset):
    print(f"Loading base model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Prepare model for training
    model.enable_input_require_grads()
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA with appropriate target modules for TinyLlama
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "v_proj", "k_proj", "o_proj"
        ],
        bias="none",
        inference_mode=False
    )
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    
    # Enable gradient computation
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    model.print_trainable_parameters()
    
    # Training arguments optimized for smaller model
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,  # Reduced for smaller model
        save_strategy="epoch",
        logging_steps=1,
        save_total_limit=1,
        optim="adamw_torch",
        fp16=False,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        report_to="none"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    return trainer

def main():
    print("Starting training process...")
    
    # Load and process data
    tokenized_dataset, tokenizer = load_and_process_data()
    
    # Setup model and trainer
    trainer = setup_model_and_trainer(tokenizer, tokenized_dataset)
    
    # Train the model
    print("Beginning training...")
    trainer.train()
    
    # Save the final model
    print("Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    print("Training complete!")

if __name__ == "__main__":
    main() 