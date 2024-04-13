from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, MambaForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-1.4b-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-1.4b-hf")
dataset = load_dataset("wikitext", split="train")
#dataset = load_from_disk("/root/autodl-tmp/Deep-Learning-Final-Project/datasets/englishquotes")["train"]
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)
lora_config =  LoraConfig(
        r=8,
        target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
        task_type="CAUSAL_LM",
        bias="none"
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="text",
)
torch.cuda.empty_cache()
trainer.train()

