from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM, TrainingArguments, DistilBertForSequenceClassification
import torch
import os
from loguru import logger

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-370m-hf")
dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")

# lora config
lora_config =  LoraConfig(
    r=64,
    target_modules=["x_proj", "embeddings", "in_proj", "out_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)

# get peft
model = get_peft_model(model, lora_config)
logger.info(f'memory footprint of model: {model.get_memory_footprint() / (1024 * 1024 * 1024)} GB')
model.print_trainable_parameters()

# dataset pre
def merge_text_field(dataset):
    dataset["finetune_field"] = dataset["question"] + dataset["answer"]
    return dataset

dataset = dataset.map(merge_text_field)

training_args = TrainingArguments(
    output_dir="./mathqa-370m",
    num_train_epochs=5,
    per_device_train_batch_size=1,
    logging_dir='./logs',
    logging_steps=1000,
    learning_rate=2e-4
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    train_dataset=dataset,
    dataset_text_field="finetune_field",
)
torch.cuda.empty_cache()
logger.info("*** start training ***")
train_result = trainer.train()
logger.info("*** finish training ***")
final_save_path = os.path.join(training_args.output_dir, "adapter_path")
trainer.save_model(final_save_path)
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
print(train_result)

