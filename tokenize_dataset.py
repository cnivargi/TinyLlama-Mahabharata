import json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Set PAD token explicitly (LLaMA models do not have a default pad token)
tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as PAD token

# Define the tokenization function
def tokenize_function(example):
    return tokenizer(
        example["prompt"],  # Tokenizing the "prompt" text
        padding="max_length",  # Ensure padding works
        truncation=True,  # Truncate if text is too long
        max_length=512  # Adjust based on model limits
    )

# Load JSONL dataset
dataset_path = "D:/datasets/ValmikiRamayan/jsonl_files/formatted_dataset.jsonl"

# Read the JSONL file and load it into Hugging Face Dataset
with open(dataset_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

dataset = Dataset.from_list(data)

# Apply tokenization
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Save tokenized dataset
tokenized_dataset.save_to_disk("D:/datasets/ValmikiRamayan/tokenized_dataset")
print("Tokenized dataset saved successfully!")