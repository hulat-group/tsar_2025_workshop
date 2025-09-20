import json
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import os
from typing import Dict, List
import logging
from datasets import load_dataset

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CEFRPromptlessDataset(Dataset):
    """Dataset para entrenamiento sin prompts - solo entrada y salida"""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Usar el mismo formato que Ettin: text con prompt y response
        # full_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{item['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{item['response']}<|eot_id|>"
        full_text = f"{item['prompt']}\n{item['response']}"
        # Tokenizar el texto completo
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Los labels son los mismos que input_ids, pero con -100 para tokens de padding
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": labels.flatten()
        }

class LlamaPromptlessTrainer:
    """Entrenador para Llama3 sin prompts estructurados"""
    
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Usando dispositivo: {self.device}")
        
    def load_dataset(self, dataset_path: str) -> List[Dict]:
        """Carga el dataset desde JSON"""
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Dataset cargado: {len(data)} ejemplos")
        return data
    
    def load_dataset_jsonl(self, dataset_path: str) -> List[Dict]:
        """Carga el dataset desde JSONL como en Ettin"""
        from datasets import load_dataset
        dataset = load_dataset("json", data_files=dataset_path)
        return dataset["train"].to_list()

    def setup_model_and_tokenizer(self):
        """Configura el modelo y tokenizer para entrenamiento promptless"""
        logger.info("Cargando modelo y tokenizer...")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Configurar tokens especiales
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Cargar modelo
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Configurar LoRA para entrenamiento eficiente
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_datasets(self, data: List[Dict], train_split: float = 0.9):
        """Prepara datasets de entrenamiento y validación"""
        # Dividir datos
        split_idx = int(len(data) * train_split)
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        
        # Crear datasets con una sola longitud máxima
        train_dataset = CEFRPromptlessDataset(train_data, self.tokenizer, max_length=512)
        val_dataset = CEFRPromptlessDataset(val_data, self.tokenizer, max_length=512)
        
        logger.info(f"Dataset de entrenamiento: {len(train_dataset)} ejemplos")
        logger.info(f"Dataset de validación: {len(val_dataset)} ejemplos")
        
        return train_dataset, val_dataset
        
    def tokenize_function(self, examples):
        """Función de tokenización para datasets - método de clase para ser serializable"""
        texts = [f"{prompt}\n{response}" 
                for prompt, response in zip(examples["prompt"], examples["response"])]
        # texts = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{response}<|eot_id|>" 
        #         for prompt, response in zip(examples["prompt"], examples["response"])]
        return self.tokenizer(texts, truncation=True, padding="max_length", max_length=512)
        
    def train(self, 
              train_path: str,
              test_path: str,
              output_dir: str = "./trained_promptless_model"):
        """Entrena el modelo sin prompts usando el patrón de Ettin"""
        
        # Cargar datos usando datasets como en Ettin
        from datasets import load_dataset
        data_files = {"train": train_path, "validation": test_path}
        dataset = load_dataset("json", data_files=data_files)
        
        # Tokenizar datasets usando el método de la clase
        tokenized_dataset = dataset.map(self.tokenize_function, batched=True, remove_columns=["prompt", "response"])
        
        # Usar los mismos argumentos que Ettin
        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            learning_rate=2e-5,
            num_train_epochs=5,
            weight_decay=0.01,
            save_total_limit=2,
            push_to_hub=False,
            report_to="none",
            fp16=True,
            bf16=False
        )
        
        # Configurar data collator
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            processing_class=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Entrenar
        logger.info("Iniciando entrenamiento...")
        trainer.train()
        
        # Guardar modelo
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Modelo guardado en: {output_dir}")

    def load_trained_model(self, model_path: str):
        """Carga el modelo entrenado desde disco"""
        logger.info(f"Cargando modelo entrenado desde: {model_path}")
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Cargar modelo
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        logger.info("Modelo cargado exitosamente")

    def test_model(self, test_examples: List[str], model_path: str = None, max_length: int = 512):
        """Prueba el modelo con ejemplos de texto"""
        if model_path:
            self.load_trained_model(model_path)
        
        logger.info("Iniciando pruebas del modelo...")
        
        for i, text in enumerate(test_examples):
            print(f"\n--- Ejemplo {i+1} ---")
            print(f"Texto original: {text}")
            
            # Formatear entrada como en entrenamiento
            input_text = f"<|user|>\nSimplify the following text to CEFR level B1{text}\n<|assistant|>\n"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=len(inputs["input_ids"][0]) + 200,
                    temperature=0.7,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True
                )
                
                # Decodificar solo la respuesta generada
                input_length = inputs["input_ids"].shape[1]
                generated_tokens = outputs[0][input_length:]
                simplified_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            print(f"Texto simplificado: {simplified_text.strip()}")

def main():
    """Función principal para entrenamiento promptless"""
    print("=== Entrenamiento Promptless de Llama3 para Simplificación ===")
    
    # Rutas de los archivos JSONL
    train_path = "./generated_cefr_dataset/cefr_generated_dataset.json"
    test_path = "./test_prompts_v4.jsonl"
    output_dir = "./llama3_promptless_finetuned_v2"
    
    # Inicializar trainer
    trainer = LlamaPromptlessTrainer()
    
    # Opción para solo entrenar o solo probar
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Modo de prueba: cargando modelo existente y probando ejemplos")
        # Solo probar modelo existente
        test_examples = [
            "Now NASA is working towards logging some of the smaller asteroids, those measuring 140 metres wide or more.",
            "The researchers found that the new treatment was significantly more effective than traditional methods.",
            "Climate change is causing unprecedented changes in weather patterns around the world."
        ]
        trainer.test_model(test_examples, model_path=output_dir)
    else:
        # Entrenar modelo
        trainer.setup_model_and_tokenizer()
        trainer.train(train_path, test_path, output_dir)
        
        # Probar modelo después del entrenamiento
        print("\n=== Probando modelo entrenado ===")
        test_examples = [
            "Now NASA is working towards logging some of the smaller asteroids, those measuring 140 metres wide or more.",
            "The researchers found that the new treatment was significantly more effective than traditional methods."
        ]
        trainer.test_model(test_examples)
    
    print("\n=== Proceso completado ===")

if __name__ == "__main__":
    main()
    
    print("\n=== Entrenamiento completado ===")


