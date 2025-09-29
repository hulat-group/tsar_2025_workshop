from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

# Ruta de los archivos JSONL
train_path = "./dataset/train_prompts_cefr_generated.jsonl"
test_path = "./dataset/test_prompts_v4.jsonl"

# Cargar dataset con Hugging Face
data_files = {"train": train_path, "validation": test_path}
dataset = load_dataset("json", data_files=data_files)

# Combinar prompt y response para entrenamiento causal, con <|im_end|>
def preprocess(example):
    return {"text": f"<|user|>\n{example['prompt']}\n<|assistant|>\n{example['response']}"}

dataset = dataset.map(preprocess, remove_columns=["prompt", "response"])

# Tokenizador y modelo base
model_id = "jhu-clsp/ettin-decoder-400m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Tokenizar texto
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True)

# Argumentos de entrenamiento
output_dir = "./finetuned/ettin_decoder_finetuned_400m"
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

# Collator para CausalLM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Entrenar
trainer.train()

# Guardar modelo
print("Guardando modelo y tokenizador en:", output_dir)
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
