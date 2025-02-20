import os
import json

# Paths
input_folder = r"D:\datasets\ValmikiRamayan\jsonl_files"  # Update if needed
output_file = os.path.join(input_folder, "merged_dataset.jsonl")

# Collect all JSONL files
jsonl_files = [f for f in os.listdir(input_folder) if f.endswith(".jsonl")]
if not jsonl_files:
    print("❌ No JSONL files found!")
    exit()

# Merge files
with open(output_file, "w", encoding="utf-8") as outfile:
    for file in jsonl_files:
        with open(os.path.join(input_folder, file), "r", encoding="utf-8") as infile:
            for line in infile:
                outfile.write(line)  # Append each line

print(f"✅ Merged {len(jsonl_files)} JSONL files into {output_file}")
