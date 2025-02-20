from datasets import load_dataset
from transformers import AutoTokenizer

# Paths
dataset_path = r"D:\datasets\ValmikiRamayan\jsonl_files\formatted_dataset.jsonl"
tokenizer_name = "meta-llama/Llama-2-7b-chat-hf"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Load dataset
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Tokenize function
def tokenize_function(examples):
    return tokenizer(examples["prompt"], padding="max_length", truncation=True)

# Apply tokenization
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Save tokenized dataset
tokenized_datasets.save_to_disk(r"D:\datasets\ValmikiRamayan\jsonl_files\tokenized_data")

print("✅ Tokenized dataset saved successfully!")
