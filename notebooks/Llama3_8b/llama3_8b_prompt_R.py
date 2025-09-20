import json
import csv
import os
import time
from typing import List, Dict, Tuple
# Reemplazar ollama por transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re

# Importar AlignScore (asumiendo que está instalado)
try:
    from AlignScore_v2.src.alignscore import AlignScore
except ImportError:
    print("Warning: AlignScore no está instalado. Instálalo con: pip install alignscore")
    AlignScore = None

class CEFRTextSimplifier:
    def __init__(self, model_path="./your_model", tokenizer_path=None, device="cuda:0"):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Cargar modelo y tokenizador local
        print(f"Cargando modelo desde: {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            print(f"Modelo cargado exitosamente en: {self.device}")
        except Exception as e:
            print(f"Error cargando el modelo: {e}")
            raise
        
        # Inicializar AlignScore
        if AlignScore:
            self.align_scorer = AlignScore(model='roberta-base', batch_size=32, device='cuda:0', ckpt_path='checkpoints/AlignScore-base.ckpt', evaluation_mode='nli_sp')
        else:
            self.align_scorer = None
        
        # Definir prompts específicos por nivel CEFR
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
        """Carga los datos de test desde el archivo JSONL"""
        test_data = []
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        test_data.append(data)
            
            print(f"Cargados {len(test_data)} elementos de test")
            return test_data
            
        except Exception as e:
            print(f"Error cargando datos de test: {e}")
            return []
    
    def extract_text_from_prompt(self, prompt: str) -> str:
        """Extrae el texto original del prompt"""
        # Buscar patrón "Simplify the following text to CEFR level X: <text>"
        pattern = r"Simplify the following text to CEFR level \w+:?\s*(.*)"
        match = re.search(pattern, prompt)
        
        if match:
            return match.group(1).strip()
        else:
            # Fallback: usar todo el texto después del ":"
            if ":" in prompt:
                return prompt.split(":", 1)[1].strip()
            return prompt
    
    def simplify_text(self, INPUT: str, target_level: str) -> str:
        """Simplifica un texto usando el modelo local"""
        

        if target_level not in self.level_prompts:
            print(f"Warning: Nivel {target_level} no reconocido, usando B1")
            target_level = "B1"

        prompt = self.level_prompts[target_level].format(INPUT=INPUT)

        try:
            # Tokenizar el prompt
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            
            # Mover inputs al dispositivo del modelo
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decodificar solo los tokens nuevos generados
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            simplified_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Limpiar respuesta de posibles artefactos
            simplified_text = self._clean_response(simplified_text)
            
            return simplified_text
            
        except Exception as e:
            print(f"Error simplificando texto: {e}")
            return INPUT  # Devolver texto original en caso de error

    def _clean_response(self, text: str) -> str:
        """Limpia la respuesta del modelo"""
        # Remover prefijos comunes
        prefixes_to_remove = [
            "Simplified text:",
            "Here is the simplified text:",
            "The simplified version is:",
            "Simplification:"
        ]
        
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remover comillas si el texto completo está entre comillas
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()
        
        return text
    
    def calculate_alignscore(self, generated_text: str, reference_text: str) -> float:
        """Calcula el AlignScore entre el texto generado y la referencia"""
        
        if not self.align_scorer:
            print("Warning: AlignScore no está disponible")
            return 0.0
        
        try:
            score = self.align_scorer.score(contexts=[generated_text], claims=[reference_text])
            return float(score[0]) if isinstance(score, list) else float(score)
        
        except Exception as e:
            print(f"Error calculando AlignScore: {e}")
            return 0.0
    
    def process_test_dataset(self, test_data: List[Dict]) -> List[Dict]:
        """Procesa todo el dataset de test"""
        
        results = []
        
        print(f"Procesando {len(test_data)} elementos del dataset...")
        
        for i, item in enumerate(tqdm(test_data, desc="Simplificando textos")):
            try:
                # Extraer información del item
                text_id = item.get('text_id', f"item_{i}")
                prompt = item.get('original', '')
                reference_text = item.get('reference', '')
                target_level = item.get('target_cefr', 'B1')
                
                # Extraer texto original del prompt usando el método específico
                original_text = prompt
                
                # Generar simplificación
                generated_text = self.simplify_text(original_text, target_level)
                
                # Calcular AlignScore
                align_score = self.calculate_alignscore(generated_text, reference_text)
                
                # Crear resultado con todos los campos necesarios
                result = {
                    'text_id': text_id,
                    'original': original_text,
                    'target_cefr': target_level,
                    'reference': reference_text,
                    'simplified': generated_text,
                    'AlignScore': align_score
                }
                
                results.append(result)
                
                # Pequeña pausa para evitar sobrecarga
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Error procesando elemento {i}: {e}")
                continue
        
        return results
    
    def save_results_to_csv(self, results: List[Dict], output_path: str):
        """Guarda los resultados en un archivo CSV"""
        
        if not results:
            print("No hay resultados para guardar")
            return
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Definir las columnas específicas solicitadas
        fieldnames = ['text_id', 'reference', 'simplified', 'AlignScore']
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                # Filtrar solo los campos requeridos para cada resultado
                for result in results:
                    filtered_result = {field: result.get(field, '') for field in fieldnames}
                    writer.writerow(filtered_result)
            
            print(f"Resultados guardados en: {output_path}")
            
            # Mostrar estadísticas básicas
            align_scores = [r['AlignScore'] for r in results if r['AlignScore'] > 0]
            if align_scores:
                avg_score = sum(align_scores) / len(align_scores)
                print(f"AlignScore promedio: {avg_score:.4f}")
                print(f"AlignScore mínimo: {min(align_scores):.4f}")
                print(f"AlignScore máximo: {max(align_scores):.4f}")
            
        except Exception as e:
            print(f"Error guardando resultados: {e}")
    
    def save_results_to_jsonl(self, results: List[Dict], output_path: str):
        """Guarda los resultados en un archivo JSONL con 4 campos específicos"""
        
        if not results:
            print("No hay resultados para guardar")
            return
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as jsonl_file:
                for result in results:
                    # Crear objeto con solo los 4 campos requeridos para JSONL
                    jsonl_record = {
                        'text_id': result.get('text_id', ''),
                        'original': result.get('original', ''),
                        'target_cefr': result.get('target_cefr', ''),
                        'reference': result.get('reference', ''),
                        'simplified': result.get('simplified', '')
                    }
                    json.dump(jsonl_record, jsonl_file, ensure_ascii=False)
                    jsonl_file.write('\n')
            
            print(f"Resultados JSONL guardados en: {output_path}")
            
        except Exception as e:
            print(f"Error guardando resultados JSONL: {e}")

def main():
    """Función principal"""
    
    # Configurar rutas
    test_data_path = "./trialdata.jsonl"
    output_csv_path = "./simplification_results_finetune_prompt_v1.csv"
    output_jsonl_path = "./simplification_results_finetune_prompt_v1.jsonl"

    model_path = "./llama3_promptless_finetuned_v1"

    # Inicializar simplificador con modelo local
    simplifier = CEFRTextSimplifier(
        model_path=model_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    
    # Cargar datos de test
    test_data = simplifier.load_test_data(test_data_path)
    
    if not test_data:
        print("No se pudieron cargar los datos de test")
        return
    
    # Procesar dataset (puedes limitar con [:10] para pruebas)
    # results = simplifier.process_test_dataset(test_data[:10])  # Para pruebas
    results = simplifier.process_test_dataset(test_data)  # Dataset completo
    
    # Guardar resultados en ambos formatos
    simplifier.save_results_to_csv(results, output_csv_path)
    simplifier.save_results_to_jsonl(results, output_jsonl_path)
    
    print("¡Proceso completado!")


if __name__ == "__main__":
    main()
