# 🧠 HULAT-UC3M @ TSAR 2025 Shared Task

[![TSAR 2025](https://img.shields.io/badge/TSAR-2025-blue)](https://tsar-workshop.github.io/shared-task/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 🌐 Official Task Website
👉 [TSAR 2025 Shared Task](https://tsar-workshop.github.io/shared-task/)

---

## 👥 Team

**HULAT-UC3M (Human Language and Accessibility Technologies)**  
Universidad Carlos III de Madrid  

- Paloma Martínez Fernández  
- Lourdes Moreno López  
- Jesús Manuel Sánchez Gómez  
- Marco Antonio Sánchez Escudero
- Javier Madrid Hijosa

---

## 📝 Approach Summary

We participated in the **TSAR 2025** task using the **Meta LLaMA 3 model (8B parameters)**, without additional fine-tuning, relying solely on **prompting strategies**.  
We submitted **two runs**:  

- **Run 1: Reinforced Prompt**  
  Detailed descriptions of each CEFR level guided the model to simplify the texts while explicitly indicating the target level.  

- **Run 2: Slightly Reinforced Prompt**  
  A shorter version of the CEFR level descriptions, aiming to balance precision and conciseness in simplification.   

---

## 🤖 Models

- **Meta LLaMA 3 – 8B**  
  - Official model: [Meta LLaMA 3 on Hugging Face](https://huggingface.co/meta-llama)  

- **Ettin – Encoder and Decoder (400M)**  
  - Official models: [Ettin Suite on Hugging Face](https://huggingface.co/collections/jhu-clsp/encoders-vs-decoders-the-ettin-suite-686303e16142257eed8e6aeb)  
⚠️ Model weights are **not included** in this repository due to size and license restrictions.  

---

## 📖 Citation (STAND BY)

@inproceedings{hulat_uc3m_tsar2025,
  author    = {Martínez, Paloma and Moreno, Lourdes and Sánchez Gómez, Jesús Manuel 
               and Madrid, Javier and Sánchez Escudero, Marco Antonio},
  title     = {HULAT-UC3M at TSAR 2025: Prompt-based Approaches with LLaMA 3 for Multilingual Text Simplification},
  booktitle = {Proceedings of the TSAR 2025 Shared Task},
  year      = {2025},
  address   = {Madrid, España},
  publisher = {}
}

---

## Funding

This work is part of the R&D&i ACCESS2MEET (PID2020-116527RB-I0) project financed by MCIN AEI/10.13039/501100011033/

