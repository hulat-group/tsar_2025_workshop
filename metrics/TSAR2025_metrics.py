# Evaluation Script for TSAR 2025 Shared Task on Readability-Controlled Text Simplification
# !pip install --upgrade huggingface_hub

# !pip install evaluate

# !pip install bert_score

from huggingface_hub import login
from sklearn.metrics import f1_score,root_mean_squared_error
from transformers import pipeline,AutoTokenizer
import evaluate
import numpy as np
import pandas as pd
import json
import csv

login()

# 1. Load Evaluation Models and Scores

# CEFR Compliance
# We will leverage predictions from three models fine-tuned on subsets of the [UniversalCEFR dataset](https://universalcefr.github.io/). Each model is based on the ModernBERT architecture and has been fine-tuned on a distinct subset of the data: (1) English document-level texts, (2) English sentence-level texts, and (3) a multilingual collection covering all available languages.

cefr_labeler1 = pipeline(task="text-classification",model="AbdullahBarayan/ModernBERT-base-doc_en-Cefr" )
cefr_labeler2 = pipeline(task="text-classification",model="AbdullahBarayan/ModernBERT-base-doc_sent_en-Cefr")
cefr_labeler3 = pipeline(task="text-classification",model="AbdullahBarayan/ModernBERT-base-reference_AllLang-Cefr")

# After obtaining a CEFR prediction from each model, we will select the prediction from the model with the highest confidence score.

def get_cefr_labels(simplifications: list, models=[cefr_labeler1,cefr_labeler2,cefr_labeler3]):
  cefr_labels = []
  for simplification in simplifications:
    top_preds = (model(simplification)[0] for model in models)
    best = max(top_preds, key=lambda d: d["score"])
    cefr_labels.append(best["label"])
  return cefr_labels

# Finally, similar to [(Barayan et al., 2025)](https://aclanthology.org/2025.coling-main.452/), CEFR compliance will be assessed based on three metrics:
# 1. weighted_f1:
# 2. adj_accuracy: measures the percentage of texts for which the system’s output aligns closely with the target CEFR level. Specifically, it considers outputs successful if their CEFR level is within one level of the specified target.
# 3. rmse: measures the average error between the estimated and target CEFR levels of the system output, by calculating the square root of the average squared differences.

CEFR_LABELS = ['A1','A2','B1','B2','C1','C2']
LABEL2IDX   = {label: idx for idx, label in enumerate(CEFR_LABELS)}

def get_cefr_compliance_score(simplifications: list, reference_levels: list, models=[cefr_labeler1,cefr_labeler2,cefr_labeler3]):

  assert len(simplifications) == len(reference_levels), "The number of simplifications is different of the number of reference_levels."

  predicted_labels = get_cefr_labels(simplifications=simplifications, models=models)
  f1 = f1_score(reference_levels, predicted_labels, average='weighted')

  true_idx = np.array([LABEL2IDX[l] for l in reference_levels])
  pred_idx =  np.array([LABEL2IDX[l] for l in predicted_labels])

  adj_acc  = (np.abs(true_idx - pred_idx) <= 1).mean()
  rmse = root_mean_squared_error(true_idx, pred_idx)

  return {'weighted_f1' : round(f1,4),
          'adj_accuracy': round(adj_acc,4),
          'rmse'        : round(rmse,4)}

# Meaning Preservation Original <> System Output, and Similarity Reference <> System Output
# We will use two metrics to compute semantic similarity: [MeaningBERT](https://doi.org/10.3389/frai.2023.1223924) (Beauchemin et al., 2023) and [BERTScore](https://openreview.net/forum?id=SkeHuCVFDr) (Zhang et al., 2020).

# MeaningBERT

meaning_bert = evaluate.load("davebulaval/meaningbert")

def get_meaningbert_score(predictions, references, model):
  assert len(predictions) == len(references), "The number of references is different of the number of predictions."
  result=[]
  for pred, ref in zip(predictions, references):
        score = model.compute(predictions=[pred], references=[ref])
        result.append(score["scores"][0])
  return round(np.mean(result)/100, 4)

def get_meaningbert_score_per_text(predictions, references, model):
  assert len(predictions) == len(references), "The number of references is different of the number of predictions."
  result = []
  for pred, ref in zip(predictions, references):
        score = model.compute(predictions=[pred], references=[ref])
        result.append(round(score["scores"][0]/100, 4))
  return result

# BERTScore

bertscore = evaluate.load("bertscore")

def get_bertscore(predictions, references, scorename=bertscore, scoretype="f1",  modelname=None):
  result = scorename.compute(references=references, predictions=predictions, lang="en") # , model_type=modelname)
  return round(np.mean(result[scoretype]), 4)

def get_bertscore_per_text(predictions, references, scorename=bertscore, scoretype="f1", modelname=None):
  result = scorename.compute(references=references, predictions=predictions, lang="en")
  return [round(score, 4) for score in result[scoretype]]

# AlignScore
try:
    from AlignScore_v2.src.alignscore import AlignScore
except ImportError:
    print("Warning: AlignScore no está instalado. Instálalo con: pip install alignscore")
    AlignScore = None
alignscore = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='checkpoints/AlignScore-base.ckpt', evaluation_mode='nli_sp')

def get_alignscore(predictions, references, scorename=alignscore):
  result = alignscore.score(contexts=predictions, claims=references)
  return [round(score, 4) for score in result]

# 2. Prepare Submission
# Submissions must be provided as `.jsonl` files. Each line in the file should contain a JSON object with the following fields:
# - text_id: the identifier of the corresponding entry in the test dataset  
# - simplified: the system-generated simplification of the entry, targeting the specified target_cefr level
# Further instructions will be provided for the final submissions once the test set is released.

# 3. Read Submission and References

def read_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

ref_data = read_jsonl('./trialdata.jsonl')
sys_data = read_jsonl('./simplification_results_promptless_v2.jsonl')

# Verify that both files have the same number of rows and matching text_ids
assert len(ref_data) == len(sys_data), "Files do not have the same number of rows."

ref_ids = [entry['text_id'] for entry in ref_data]
sys_ids = [entry['text_id'] for entry in sys_data]

assert ref_ids == sys_ids, "Mismatch in text_id order or values."

# Create lists with the necessary information for the metrics
original_texts = [entry['original'] for entry in ref_data]
target_cefr_levels = [entry['target_cefr'].upper() for entry in ref_data]
reference_texts = [entry['reference'] for entry in ref_data]
simplified_texts = [entry['simplified'] for entry in sys_data]

# 3. Evaluate Submission

# Get per-text scores
predicted_cefr_labels = get_cefr_labels(simplified_texts, [cefr_labeler1,cefr_labeler2,cefr_labeler3])
alignscore_scores = get_alignscore(simplified_texts, reference_texts, alignscore)
meaningbert_scores_ref = get_meaningbert_score_per_text(simplified_texts, reference_texts, meaning_bert)
bertscore_scores_ref = get_bertscore_per_text(simplified_texts, reference_texts, bertscore)

# Create CSV with per-text results
csv_data = []
for i in range(len(ref_data)):
    csv_data.append({
        'text_id': ref_data[i]['text_id'],
        'AlignScore': alignscore_scores[i],
        'CEFR_pred': predicted_cefr_labels[i],
        'CEFR_target': target_cefr_levels[i],
        'MeaningBERT_Ref': meaningbert_scores_ref[i],
        'BERTScore_Ref': bertscore_scores_ref[i]
    })

# Save to CSV
csv_filename = 'tsar2025_detailed_results_promptless_v2.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['text_id', 'AlignScore', 'CEFR_pred', 'CEFR_target', 'MeaningBERT_Ref', 'BERTScore_Ref']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Detailed results saved to {csv_filename}")

# CEFR Compliance (aggregated)
compliance_score = get_cefr_compliance_score(simplified_texts, target_cefr_levels, [cefr_labeler1,cefr_labeler2,cefr_labeler3])

# Meaning Preservation between Original and System Output
meaningbert_score_org = get_meaningbert_score(simplified_texts, original_texts, meaning_bert)
bertscore_org = get_bertscore(simplified_texts, original_texts, bertscore)

# Similarity between System Output and References
meaningbert_score_ref = get_meaningbert_score(simplified_texts, reference_texts, meaning_bert)
bertscore_ref = get_bertscore(simplified_texts, reference_texts, bertscore)

# Overview Scores
result_dict = {
    "CEFR Compliance": {
        "weighted_f1": compliance_score["weighted_f1"],
        "adj_accuracy": compliance_score["adj_accuracy"],
        "rmse": compliance_score["rmse"]
    },
    "Meaning Preservation": {
        "MeaningBERT-Orig": meaningbert_score_org,
        "BERTScore-Orig": bertscore_org
    },
    "Similarity to References": {
        "MeaningBERT-Ref": meaningbert_score_ref,
        "BERTScore-Ref": bertscore_ref
    }
}

print(result_dict)

