from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

splits = ["train", "test"]
cols_to_keep = ["prompt", "prompt_label"]
ds = {
    split: ds
    for split, ds in zip(
        splits,
        load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=splits),
    )
}

for split in splits:
    ds[split] = ds[split].shuffle(seed=42).select(range(500))
    ds[split] = ds[split].select_columns(cols_to_keep)

model = AutoModelForSequenceClassification.from_pretrained(
    "gpt2",
    num_labels=2,
    id2label={0: "safe", 1: "unsafe"},
    label2id={"safe": 0, "unsafe": 1},
).to(device)

for param in model.base_model.parameters():
    param.requires_grad = False

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = (
    tokenizer.eos_token
)  # because gpt2 tokenizer needs a pad token to be specified
model.config.pad_token_id = tokenizer.pad_token_id


def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples["prompt"], padding="max_length", truncation=True
    )
    label_map = {"safe": 0, "unsafe": 1}
    tokenized_examples["labels"] = [
        label_map[label] for label in examples["prompt_label"]
    ]
    return tokenized_examples


tokenized_ds = {}
for split in splits:
    tokenized_ds[split] = ds[split].map(preprocess_function, batched=True)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}


trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./data/prompt_analysis",
        learning_rate=2e-3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# without training, evaluate
# trainer.train()
print(trainer.evaluate())
