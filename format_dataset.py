import json
import os

# Input & Output Paths
input_file = r"D:\datasets\ValmikiRamayan\jsonl_files\merged_dataset.jsonl"
output_file = r"D:\datasets\ValmikiRamayan\jsonl_files\formatted_dataset.jsonl"

# Load dataset
formatted_data = []
with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            data = json.loads(line)  # Read JSONL line by line
            
            # Extract Sanskrit text
            if "content" in data:
                verse = data["content"]  
            else:
                print(f"⚠️ Skipping entry, no 'content' field found: {data}")
                continue
            
            # Optional: Include translation as context
            explanation = data.get("explanation", "")

            # Convert to prompt-response format
            formatted_entry = {
                "prompt": f"{verse}\n\nQ: What is the meaning of this verse?",
                "response": explanation if explanation else "This is a verse from the Ramayan in Sanskrit."
            }
            formatted_data.append(formatted_entry)

        except json.JSONDecodeError:
            print(f"❌ Error reading line: {line}")

# Save formatted dataset
with open(output_file, "w", encoding="utf-8") as outfile:
    for entry in formatted_data:
        outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Formatted dataset saved at {output_file} with {len(formatted_data)} entries.")
