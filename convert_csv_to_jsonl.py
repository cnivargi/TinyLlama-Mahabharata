import os
import pandas as pd
import json

# Set dataset folder
dataset_folder = r"D:\Datasets\ValmikiRamayan"  # Update if needed
output_folder = os.path.join(dataset_folder, "jsonl_files")

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Check if CSV files exist
csv_files = [f for f in os.listdir(dataset_folder) if f.endswith(".csv")]
if not csv_files:
    print("❌ No CSV files found in", dataset_folder)
else:
    print("✅ Found CSV files:", csv_files)

# Function to convert CSV to JSONL
def convert_csv_to_jsonl(csv_file, jsonl_file):
    try:
        df = pd.read_csv(csv_file)  # Load CSV
        df = df.dropna()  # Remove empty rows
        with open(jsonl_file, "w", encoding="utf-8") as jsonl_out:
            for record in df.to_dict(orient="records"):
                jsonl_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"✔ Successfully converted: {csv_file} → {jsonl_file}")
    except Exception as e:
        print(f"❌ Error processing {csv_file}: {e}")

# Convert each CSV to JSONL
for file in csv_files:
    csv_path = os.path.join(dataset_folder, file)
    jsonl_path = os.path.join(output_folder, file.replace(".csv", ".jsonl"))
    convert_csv_to_jsonl(csv_path, jsonl_path)

print("✅ Conversion complete! JSONL files saved in:", output_folder)