import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from bitsandbytes.optim import Adam8bit

# Define dataset path
dataset_path = r"D:\new_dataset\data-00000-of-00001.arrow"  # Or the full path to the .arrow file

# Load dataset
print("Trying to load file:", dataset_path)
try:
    dataset = load_dataset("arrow", data_files=dataset_path)
    print("Dataset loaded successfully!\n", dataset)

    if dataset is None or "train" not in dataset:
        raise ValueError("Error: Dataset does not contain a 'train' split.")

except FileNotFoundError:
    print(f"Error: File not found at {dataset_path}")
    exit(1)
except Exception as e:
    print(f"An error occurred during dataset loading: {e}")
    exit(1)

# Extract the train dataset
train_dataset = dataset["train"]
print("\nTrain dataset summary:", train_dataset)

# Load pre-trained model & tokenizer with quantization
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Padding token set to: {tokenizer.pad_token}")

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")  # Load in 8-bit
model.resize_token_embeddings(len(tokenizer)) # Resize if you add a new token

# Tokenize dataset
def tokenize_function(example):
    encodings = tokenizer(
        example["prompt"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    encodings["labels"] = encodings["input_ids"][:]  # Set labels = input_ids for CLM
    return encodings

train_dataset = train_dataset.map(tokenize_function, batched=True)

# Convert dataset to PyTorch tensors
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments (adjust these)
training_args = TrainingArguments(
    output_dir="D:/GenAI2025/FineTune/final_model",
    num_train_epochs=3,  # Or more if needed
    per_device_train_batch_size=1,  # Minimum batch size
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=10,
    logging_strategy="steps",
    save_total_limit=2,
    report_to="tensorboard",  # Or remove for basic logging
    fp16=True,  # Use mixed precision (if your GPU supports it)
    warmup_steps=500,  # Adjust as needed
    weight_decay=0.01,  # Adjust as needed
    learning_rate=5e-5,  # Adjust as needed
    optim="paged_adam8bit",  # Use the 8-bit optimizer
)

# Optimizer
optimizer = Adam8bit(model.parameters(), lr=training_args.learning_rate)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    optimizers=(optimizer, None),
)

# Training
print("Starting training...")
trainer.train()
print("Fine-tuning complete! Model saved to: D:/GenAI2025/FineTune/final_model")