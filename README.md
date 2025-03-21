# Udacity GenAI NanoDegree Project 1

- PEFT Technique : LoRA, as recommended
- Model : gpt2, as recommended
- Evaluation approach: HuggingFace Trainer with accuracy as the base metric for evaluation
- Fine-tuning dataset: [AI Content Safety Dataset by Nvidia](nvidia/Aegis-AI-Content-Safety-Dataset-2.0) The full dataset contains prompts to an LLM and respective responses. It also has labels for prompts and responses as to whether they are 'safe' or 'unsafe'. For this project, I've focused on only prompts since the human variety of expression will be harder to classify


# Scripts

- `base_gpt2.py` - Loads pretrained `gpt2` model, tokenizes the data and then without any training, evaluates the model on the test set
- `peft_gpt2.py` - Loads pretrained `gpt2` model, does `LoRA` technique of `PEFT`, trains the model for 2 epochs and evaluates the test set

# Results

- The base GPT2 model accuracy on the evaluation set is ~53.2%
- The PEFT LORA GPT2 model accuracy on the evaluation set is ~77%


# Note

The workspace package versions would not work for me with the error when trying to train the PEFT GPT2 model.
```
ValueError: You should supply an encoding or a list of encodings to this method that includes input_ids, but you provided ['labels']
```

On my computer with the packages as specified in the `environment.yml` file, the code works without any issues
