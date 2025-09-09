# 📊 Results

This folder contains the outputs and evaluation metrics of our runs for the **TSAR 2025 Shared Task**.  
It is divided into two main components: **simplifications** (system outputs) and **metrics** (quantitative evaluation).

---

## 📝 Simplifications
(ACTUALIZAR DE ACUERDO A LO NOMBRES...)
- `trial_simplifications.jsonl` → system outputs for the **trial set**.  
- `test_simplifications.jsonl` → system outputs for the **test set**.  
  
  ```json
  {
    "text_id": "number-a1/a2",
    "original": "...",
    "target_cefr": "A1/A2/B1",
    "reference": "...",
    "simplified": "..."
  }

- Additional runs included in [`../runs/`](../runs/).

Each file follows the shared task format:  

  ```json
  {
    "text_id": "number-a1/a2",
    "simplified": "...",
    "target_cefr": "A1/A2/B1"
  }

---

## 📐 Metrics

We report evaluation results across several metrics:  

- **CEFR Compliance**: Weighted F1, Adjacent Accuracy, Exact match, RMSE  
- **Meaning Preservation**: MeaningBERT, BERTScore  
- **Similarity to References**: MeaningBERT, BERTScore  
- **AlignScore**  

### 📊 Summary Table

| Metric                                 | LLaMA 3 8B (Reinforced)  | LLaMA 3 8B (Slightly Reinforced) | Ettin Decoder |
|----------------------------------------|--------------------------|----------------------------------|---------------|
| CEFR Compliance – Weighted F1*         | 0.3000                   | 0.5200                           | 0.4800        |
| CEFR Compliance – Adjacent Accuracy*   | 0.8500                   | 0.9750                           | 0.9300        |
| CEFR Compliance – Exact                | 0.1750                   | 0.4750                           | –             |
| CEFR Compliance – RMSE*                | 1.1100                   | 0.7746                           | 0.8900        |
| Meaning Preservation – MeaningBERT*    | 0.6532                   | 0.7170                           | 0.6901        |
| Meaning Preservation – BERTScore*      | 0.8837                   | 0.8988                           | 0.9025        |
| Similarity to Refs – MeaningBERT*      | 0.6384                   | 0.7075                           | 0.6243        |
| Similarity to Refs – BERTScore*        | 0.8764                   | 0.8921                           | 0.8789        |
| AlignScore                             | 0.5600                   | 0.6038                           | 0.4300        |

---

## 📂 Files Included (REVISAR)

- `trial_metrics.xlsx` → results on the trial dataset.  
- `test_metrics.xlsx` → results on the test dataset.  
- `comparison.xlsx` → consolidated comparison across models.  
- `*.csv` files with raw metric outputs.  

---

✨ These results demonstrate the effectiveness of prompt-based approaches with **LLaMA 3 8B** and provide a baseline comparison with the **Ettin Decoder**.
