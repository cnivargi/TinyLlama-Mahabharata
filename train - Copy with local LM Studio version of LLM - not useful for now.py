import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk

# Set your local model path
local_model_path = r"C:/Users/Chittaranjan/.ollama/models/blobs/RichardErkhov/meta-llama_-_Llama-2-7b-chat-hf-gguf/Llama-2-7b-chat-hf.Q4_0.gguf"

# Load the model and tokenizer from the local directory
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(local_model_path)

# Load the tokenized dataset
dataset_path = r"D:/datasets/ValmikiRamayan/tokenized_dataset"
print(f"Loading dataset from {dataset_path}...")
train_dataset = load_from_disk(dataset_path)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,  # Adjust based on GPU memory
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    logging_steps=10,
    num_train_epochs=3,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),  # Use FP16 if a GPU is available
    load_best_model_at_end=True,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Start training
print("Starting fine-tuning...")
trainer.train()

# Save the final fine-tuned model
output_model_dir = "./fine_tuned_model"
print(f"Saving fine-tuned model to {output_model_dir}...")
trainer.save_model(output_model_dir)
tokenizer.save_pretrained(output_model_dir)

print("Fine-tuning completed successfully!")
