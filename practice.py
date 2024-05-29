from transformers import (RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer,
    TrainingArguments)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tensorboard import program
import sys
import os
import yaml
import numpy as np

"""Import PolymerSmilesTokenizer from PolymerSmilesTokenization.py"""
from PolymerSmilesTokenization import PolymerSmilesTokenizer

"""Import LoadPretrainData"""
from dataset import LoadPretrainData

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
from omegaconf import OmegaConf

"""train-validation split"""
def split(file_path):
    dataset = pd.read_csv(file_path, header=None).values
    train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=1)
    return train_data, valid_data

np.random.seed(seed=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available() #checking if CUDA + Colab GPU works

config = OmegaConf.load('practice_config.yaml')

"""Set tokenizer"""
#tokenizer = RobertaTokenizer.from_pretrained("roberta-base",max_len=512)
tokenizer = PolymerSmilesTokenizer.from_pretrained("roberta-base", max_len=config.blocksize)

"""Load Data"""
train_data, valid_data = split(config.file_path)
data_train = LoadPretrainData(tokenizer=tokenizer, dataset=train_data, blocksize=config.blocksize)
data_valid = LoadPretrainData(tokenizer=tokenizer, dataset=valid_data, blocksize=config.blocksize)

"""Set DataCollator"""
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=config.mlm_probability
)


"""Training Arguments"""

training_args = TrainingArguments(
    output_dir=config.save_path,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    learning_rate=config.lr_rate,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    num_train_epochs=config.epochs,
    weight_decay=config.weight_decay,
    save_total_limit=config.save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device) # gpt2

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=data_train,
    eval_dataset=data_valid
)

trainer.train(resume_from_checkpoint=config.load_checkpoint)
trainer.save_model(config.save_path)

