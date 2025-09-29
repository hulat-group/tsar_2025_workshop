# 📓 Notebooks

This folder contains the **final notebooks** used in our participation in the **TSAR 2025 Shared Task**.  
Each notebook documents the workflow of applying the selected models to the **trial** and **test** datasets, including simplification outputs and metric evaluation.

---

## 🔹 Models Used

- **Meta LLaMA 3 – 8B parameters**  
  - Used for both trial and test runs with two prompting strategies: *Reinforced* and *Slightly Reinforced*.  
  - Official model: [Meta LLaMA 3 on Hugging Face](https://huggingface.co/meta-llama)  

- **Ettin – Decoder (400M)**  
  - Used as an additional baseline model for simplification.  
  - Official models: [Ettin Suite on Hugging Face](https://huggingface.co/collections/jhu-clsp/encoders-vs-decoders-the-ettin-suite-686303e16142257eed8e6aeb)  

⚠️ Model weights are **not included** in this repository due to size restrictions.  
Instead, notebooks show how they were loaded and used for inference and evaluation.

---

## 📂 Available Notebooks 

- **LLama3_8b** 
- `llama3_8b_train.py` → simplifications and evaluation for **trial** set using LLaMA 3 8B.  
- `llama3_8b_trial.py` → simplifications and evaluation for **trial** set using LLaMA 3 8B.  
- `llama3_8b_test.py` → simplifications and evaluation for **test** set using LLaMA 3 8B. 

- **Ettin - Decoder (400)** 
- `ettin_trial.ipynb` → simplifications and evaluation for **trial** set using Ettin Decoder.  
- `ettin_test.ipynb` → simplifications and evaluation for **test** set using Ettin Decoder.  

---

## ⚙️ How to Use

1. Open a notebook in Google Colab or Jupyter.  
2. Ensure you have the required dependencies installed:  
   ```bash
   pip install -r requirements.txt

