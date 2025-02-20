# **📖 Fine-Tuning TinyLlama on Mahabharata Dataset**  

🚀 **Fine-tuning the TinyLlama 1.1B model on the Mahabharata dataset** to generate responses based on Sanskrit literature.  
This project leverages **Unsloth, LoRA fine-tuning, and Transformers** to fine-tune a lightweight LLM within limited GPU resources.  

---

## **📌 Project Overview**  
- **Model Used**: [TinyLlama 1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B)  
- **Dataset**: [Mahabharata dataset]  
- **Fine-Tuning Framework**: 🤗 **Hugging Face Transformers + Unsloth**  
- **Training Method**: **LoRA (Low-Rank Adaptation) Fine-Tuning**  
- **Hardware Used**:  
  - Intel **i7-13500H**  
  - **16GB RAM**  
  - NVIDIA RTX **3050 (6GB VRAM)**  
  - Running on **WSL2 (Ubuntu)**  

---

## **🛠️ Installation & Setup**  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your_username/TinyLlama-Mahabharata.git
cd TinyLlama-Mahabharata
```

### **2️⃣ Set Up Virtual Environment**  
```bash
python3 -m venv venv
source venv/bin/activate   # For Linux/macOS
venv\Scripts\activate      # For Windows (PowerShell)
```

### **3️⃣ Install Dependencies**  
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
pip install transformers datasets accelerate huggingface_hub
```

---

## **📂 Project Structure**  

```
TinyLlama-Mahabharata/
│── Mahabharat/                        # Dataset (Extracted from Kaggle)
│── final_model_mahabharata/           # Fine-tuned model output
│── tokenized_mahabharata/             # Tokenized dataset
│── scripts/
│   │── preprocess.py                   # Preprocessing script
│   │── tokenize_data.py                 # Tokenization script
│   │── finetune.py                      # Fine-tuning script
│   │── chatbot.py                        # Chatbot for inference
│   │── hf_push.py                        # Upload model to Hugging Face
│── README.md                            # Project documentation
│── .gitignore                           # Ignore unnecessary files
```

---

## **📊 Dataset Preprocessing**  
We used a **text-based Mahabharata dataset**, converted it into structured tokenized format using **Hugging Face Datasets API**.

Run the preprocessing script:  
```bash
python scripts/preprocess.py
```

Tokenize the dataset:  
```bash
python scripts/tokenize_data.py
```

---

## **🦥 Fine-Tuning the Model with Unsloth**  
TinyLlama is fine-tuned using **LoRA (Low-Rank Adaptation)** to optimize performance within **limited GPU resources**.  
```bash
python scripts/finetune.py
```
💡 **This takes around 30-40 minutes on an RTX 3050.**  

---

## **🤖 Running the Fine-Tuned Model (Chatbot)**  
After fine-tuning, you can interact with the model using the chatbot script:  
```bash
python scripts/chatbot.py
```

Example Interaction:  
```bash
User: Who was Arjuna in Mahabharata?
Bot: Arjuna was one of the Pandavas, a great warrior and an archer, known for his role in the Kurukshetra war.
```

---

## **🚀 Uploading the Fine-Tuned Model to Hugging Face**  
To share your model with the community:  
```bash
python scripts/hf_push.py
```

---

## **🔧 Troubleshooting & Issues Faced**  
- **Issue: "Failed to find CUDA"**  
  - Fixed by ensuring the correct **CUDA Toolkit & PyTorch version** were installed.  
- **Issue: GitHub not allowing files larger than 100MB**  
  - Fixed by using **`.gitignore`** to prevent pushing large model weights.  
- **Issue: Missing Python files after Git push**  
  - Files were **accidentally deleted**, but recovered from **Recycle Bin**.  
- **Issue: Hugging Face API errors**  
  - Fixed by **authenticating with a valid HF token** and setting the correct `base_model`.  

---

## **📌 Future Improvements**  
✔ **Fine-tune on the Ramayana dataset** to expand knowledge coverage.  
✔ **Optimize inference speed** using **quantization** (GGML/GPTQ).  
✔ **Experiment with larger models (e.g., Mistral 7B) for better accuracy**.  

---

## **👨‍💻 Contributors**  
🚀 **@cnivargi** – Model fine-tuning & dataset preparation  
🚀 **[Chittaranjan G Nivargi]** – Project documentation & optimization  

---

## **📜 License**  
This project is licensed under the **Apache 2.0 License** – Feel free to use and modify!  

---

🔥 **Try it out and let us know your feedback!** 🚀  
🔗 **GitHub:** [https://github.com/your_username/TinyLlama-Mahabharata](https://github.com/your_username/TinyLlama-Mahabharata)  
🔗 **Hugging Face Model:** [https://huggingface.co/your_username/tinyllama-mahabharata](https://huggingface.co/your_username/tinyllama-mahabharata)  

---
