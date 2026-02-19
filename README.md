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

## 📖 Citation

For citing the GitHub repository:

@misc{sanchezgomez2025tsar2025workshop,  
      title={{HULAT-UC3M at TSAR 2025 Shared Task}},  
      author={Sanchez-Gomez, Jesus M. and Moreno, Lourdes and Mart{\'\i}nez, Paloma and Sanchez-Escudero, Marco Antonio},  
      year={2025},  
      url={https://github.com/hulat-group/tsar_2025_workshop},  
}

For citing the conference paper:

@inproceedings{sanchezgomez2025hulat,  
  title={{HULAT-UC3M at TSAR 2025 Shared Task: A Prompt-Based Approach using Lightweight Language Models for Readability-Controlled Text Simplification}},  
  author={Sanchez-Gomez, Jesus M. and Moreno, Lourdes and Mart{\'\i}nez, Paloma and Sanchez-Escudero, Marco Antonio},  
  booktitle={{Proceedings of the Fourth Workshop on Text Simplification, Accessibility and Readability (TSAR 2025)}},  
  pages={183--192},  
  publisher={{Association for Computational Linguistics}},  
  year={2025},  
  doi={10.18653/v1/2025.tsar-1.15}  
}

---

## Funding

This work has been supported by grant PID2023-148577OB-C21 (Human-Centered AI: User-Driven Adapted Language Models-HUMAN\_AI) by MICIU/AEI/10.13039/501100011033 and by FEDER/UE.

