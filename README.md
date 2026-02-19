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

@misc{moreno2025promptbasedsimplificationplainlanguage,
      title={Prompt-Based Simplification for Plain Language using Spanish Language Models}, 
      author={Lourdes Moreno and Jesus M. Sanchez-Gomez and Marco Antonio Sanchez-Escudero and Paloma Martínez},
      year={2025},
      eprint={2509.17209},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.17209}, 
}

---

## Funding

This work has been supported by grant PID2023-148577OB-C21 (Human-Centered AI: User-Driven Adapted Language Models-HUMAN\_AI) by MICIU/AEI/10.13039/501100011033 and by FEDER/UE.

