import json
import csv
import os
import time
from typing import List, Dict
from tqdm import tqdm
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Importar AlignScore (asumiendo que está instalado)
try:
    from alignscore.alignscore import AlignScore
except ImportError:
    print("Warning: AlignScore no está instalado. Instálalo con: pip install alignscore")
    AlignScore = None

class CEFRTextSimplifier:
    def __init__(self, model_name="./finetuned/ettin_decoder_finetuned_400m", device="cuda:0"):#jhu-clsp/ettin-decoder-400m
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"

        # Cargar modelo y tokenizador desde Hugging Face
        print(f"Cargando modelo {self.model_name} en {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()
        print("Modelo cargado con éxito.")

        # Inicializar AlignScore
        if AlignScore:
            self.align_scorer = AlignScore(
                model='roberta-base',
                batch_size=32,
                device='cuda:0',
                ckpt_path='./AlignScore_v2/checkpoints1/AlignScore-base.ckpt',
                evaluation_mode='nli_sp'
            )
        else:
            self.align_scorer = None

        # Prompts por nivel CEFR
        self.level_prompts = {
            "a1": """Below are CEFR descriptions:
A1: Beginner – Simple sentences. No passive. Top 1000 words. Familiar names, cognates.
A2: Elementary – Short, simple texts with predictable info. 1000–2000 words.
B1: Intermediate – Everyday or job-related language. Includes descriptions of events, feelings and wishes.
B2: Upper Intermediate – Contemporary articles, styles, viewpoints. 5000–10,000 words.
C1: Proficient – Long, complex factual/literary texts. 10,000–20,000 words.
C2: Advanced Proficient – All forms of written language including abstract and highly complex.

Example A1:
Original: Some asteroids are very small.
Simplified: Some space rocks are very small.

Target CEFR level: A1
Now simplify this one:
Original: {INPUT}""",

            "a2": """Below are CEFR descriptions:
A1: Beginner – Simple sentences. No passive. Top 1000 words. Familiar names, cognates.
A2: Elementary – Short, simple texts with predictable info. 1000–2000 words.
B1: Intermediate – Everyday or job-related language. Includes descriptions of events, feelings and wishes.
B2: Upper Intermediate – Contemporary articles, styles, viewpoints. 5000–10,000 words.
C1: Proficient – Long, complex factual/literary texts. 10,000–20,000 words.
C2: Advanced Proficient – All forms of written language including abstract and highly complex.

Example A2:
Original: Earthquakes damage buildings and bridges.
Simplified: Earthquakes can break buildings and bridges.

Target CEFR level: A2
Now simplify this one:
Original: {INPUT}""",

            "b1": """You are an expert text simplifier. Simplify the following text to CEFR B1 level.

B1 level requirements:
- Use intermediate vocabulary (most common 3000 words)
- Use various tenses but keep grammar straightforward
- Use medium-length sentences (10-15 words)
- Use common connectors and linking words
- Maintain main ideas but simplify complex concepts
- Use clear, direct language

Text to simplify: {INPUT}

Simplified text:""",

            "b2": """Below are CEFR descriptions:
A1: Beginner – Simple sentences. No passive. Top 1000 words. Familiar names, cognates.
A2: Elementary – Short, simple texts with predictable info. 1000–2000 words.
B1: Intermediate – Everyday or job-related language. Includes descriptions of events, feelings and wishes.
B2: Upper Intermediate – Contemporary articles, styles, viewpoints. 5000–10,000 words.
C1: Proficient – Long, complex factual/literary texts. 10,000–20,000 words.
C2: Advanced Proficient – All forms of written language including abstract and highly complex.

Example B1:
Original: Many wild animals are starting to enter cities.
Simplified: Some wild animals are visiting cities now.

Target CEFR level: B1
Now simplify this one:
Original: {INPUT}""",
        }

    def load_test_data(self, json_path: str) -> List[Dict]:
        """Carga los datos de test desde JSONL"""
        test_data = []
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))
            print(f"Cargados {len(test_data)} ejemplos de test")
            return test_data
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return []

    def simplify_text(self, INPUT: str, target_level: str) -> str:
        """Genera simplificación con Ettin usando prompts"""
        if target_level not in self.level_prompts:
            print(f"Warning: nivel {target_level} no reconocido, usando B1")
            target_level = "b1"

        prompt = self.level_prompts[target_level].format(INPUT=INPUT)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            return self._clean_response(generated_text)
        except Exception as e:
            print(f"Error simplificando: {e}")
            return INPUT

    def _clean_response(self, text: str) -> str:
        """Limpia artefactos del modelo"""
        prefixes = [
            "Simplified text:",
            "Here is the simplified text:",
            "The simplified version is:",
            "Simplification:"
        ]
        for p in prefixes:
            if text.startswith(p):
                text = text[len(p):].strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        return text

    def calculate_alignscore(self, generated_text: str, reference_text: str) -> float:
        """Calcula AlignScore"""
        if not self.align_scorer:
            print("Warning: AlignScore no está disponible")
            return 0.0
        try:
            score = self.align_scorer.score(contexts=[generated_text],claims=[reference_text])
            return float(score[0]) if isinstance(score, list) else float(score)

        except Exception as e:
            print(f"Error AlignScore: {e}")
            return 0.0

    def process_test_dataset(self, test_data: List[Dict]) -> List[Dict]:
        """Procesa el dataset completo"""
        results = []

        print(f"Procesando {len(test_data)} elementos del dataset...")

        for i, item in enumerate(tqdm(test_data, desc="Simplificando textos")):
            try:
                # Extraer información del item
                text_id = item.get("text_id", f"item_{i}")
                original_text = item.get("original", "")
                reference_text = item.get("reference", "")
                target_level = item.get("target_cefr", "b1").lower()

                # Generar simplificación y calcular AlignScore
                simplified = self.simplify_text(original_text, target_level)
                align_score = self.calculate_alignscore(simplified, reference_text)

                # Crear resultado con todos los campos necesarios
                results.append({
                    "text_id": text_id,
                    "original": original_text,
                    "target_cefr": target_level,
                    "reference": reference_text,
                    "simplified": simplified,
                    "AlignScore": align_score
                })

                time.sleep(0.05)  # evitar sobrecarga
            except Exception as e:
                print(f"Error en item {i}: {e}")
                continue
        return results

    def save_results_to_csv(self, results: List[Dict], path: str):
        if not results:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["text_id","original","simplified","AlignScore"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "text_id": r["text_id"],
                    "original": r["original"],
                    "simplified": r["simplified"],
                    "AlignScore": r["AlignScore"]
                })
        print(f"Resultados guardados en {path}")

    def save_results_to_jsonl(self, results: List[Dict], path: str):
        if not results:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Resultados JSONL guardados en {path}")


def main():
    test_path = "./dataset/tsar2025_trialdata.jsonl" 
    out_csv = "./results/simplification_results_ettin_FR.csv" 
    out_jsonl = "./results/simplification_results_ettin_FR.jsonl"

    simplifier = CEFRTextSimplifier()
    data = simplifier.load_test_data(test_path)
    if not data:
        return

    results = simplifier.process_test_dataset(data)
    simplifier.save_results_to_csv(results, out_csv)
    simplifier.save_results_to_jsonl(results, out_jsonl)

    print("¡Proceso completado!")


if __name__ == "__main__":
    main()
