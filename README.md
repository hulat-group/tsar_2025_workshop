# 🧠 HULAT-UC3M @ TSAR 2025 Shared Task

[![TSAR 2025](https://img.shields.io/badge/TSAR-2025-blue)](https://tsar-workshop.github.io/shared-task/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)]()

---

## 🌐 Enlace oficial de la tarea
👉 [TSAR 2025 Shared Task](https://tsar-workshop.github.io/shared-task/)

---

## 👥 Equipo

**HULAT-UC3M (Human Language and Accessibility Technologies)**  
Universidad Carlos III de Madrid  

- Paloma Martínez Fernández  
- Lourdes Moreno López  
- Jesús Manuel Sánchez Gómez  
- Javier Madrid  
- Marco Antonio Sánchez Escudero  

---

## 📝 Resumen del enfoque

Participamos en la tarea **TSAR 2025** utilizando el modelo **Meta LLaMA 3 (8B parámetros)**, sin fine-tuning adicional, únicamente mediante **estrategias de prompting**.  
Presentamos **dos ejecuciones**:  

- **Run 1: Prompt reforzado**  
  Descripciones detalladas de cada nivel CEFR, guían al modelo a simplificar los textos indicando explícitamente el nivel de destino.  

- **Run 2: Prompt ligeramente reforzado**  
  Versión más breve de las descripciones CEFR, buscando balance entre precisión y concisión en la simplificación.  

---

## ⚙️ Uso

```bash
# Ejemplo de inferencia (mock o real si se libera código)
python src/inference.py \
  --config configs/run1.yaml \
  --input data/test.jsonl \
  --output runs/run1_predictions.jsonl

# Evaluación (métricas SARI, BERTScore, FHRI, SAS…)
python src/evaluate.py \
  --pred runs/run1_predictions.jsonl \
  --refs data/references.jsonl \
  --output results/metrics_run1.csv
