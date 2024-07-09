import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Limit to a single GPU

from transformers import GPT2Config, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

# Import PolymerSmilesTokenizer and LoadPretrainData
from PolymerSmilesTokenization import PolymerSmilesTokenizer
from dataset import LoadPretrainData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train-validation split function
def split(file_path):
    dataset = pd.read_csv(file_path, header=None).values
    train_data, valid_data = train_test_split(dataset, test_size=0.2, random_state=1)
    return train_data, valid_data

def main(pretrain_config):
    # GPT-2 configuration
    config = GPT2Config(
        vocab_size=50257,  # GPT-2 default vocabulary size
        n_positions=pretrain_config['max_position_embeddings'],
        n_ctx=pretrain_config['max_position_embeddings'],
        n_embd=768,  # Default embedding size for GPT-2
        n_layer=pretrain_config['num_hidden_layers'],
        n_head=pretrain_config['num_attention_heads'],
        resid_pdrop=pretrain_config['hidden_dropout_prob'],
        embd_pdrop=pretrain_config['hidden_dropout_prob'],
        attn_pdrop=pretrain_config['attention_probs_dropout_prob'],
    )

    # Set tokenizer
    tokenizer = PolymerSmilesTokenizer.from_pretrained("gpt2", max_len=pretrain_config['blocksize'])

    # Construct Language Model
    model = GPT2LMHeadModel(config=config).to(device)

    # Load Data
    train_data, valid_data = split(pretrain_config['file_path'])
    data_train = LoadPretrainData(tokenizer=tokenizer, dataset=train_data, blocksize=pretrain_config['blocksize'])
    data_valid = LoadPretrainData(tokenizer=tokenizer, dataset=valid_data, blocksize=pretrain_config['blocksize'])
    # Set DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    

    # Training arguments
    training_args = TrainingArguments(
        output_dir=pretrain_config['save_path'],
        overwrite_output_dir=pretrain_config['overwrite_output_dir'],
        num_train_epochs=pretrain_config['epochs'],
        per_device_train_batch_size=pretrain_config['batch_size'],
        per_device_eval_batch_size=pretrain_config['batch_size'],
        save_strategy=pretrain_config['save_strategy'],
        save_total_limit=pretrain_config['save_total_limit'],
        fp16=pretrain_config['fp16'],
        logging_strategy=pretrain_config['logging_strategy'],
        evaluation_strategy=pretrain_config['evaluation_strategy'],
        learning_rate=pretrain_config['lr_rate'],
        lr_scheduler_type=pretrain_config['scheduler_type'],
        weight_decay=pretrain_config['weight_decay'],
        warmup_ratio=pretrain_config['warmup_ratio'],
        report_to=pretrain_config['report_to'],
        dataloader_num_workers=pretrain_config['dataloader_num_workers'],
        sharded_ddp=pretrain_config['sharded_ddp'],
        optim='adamw_torch',  # Use PyTorch's AdamW implementation
    )

    # Ensure CUDA is initialized before training
    if torch.cuda.is_available():
        torch.cuda.init()

    import pdb ;pdb.set_trace()
    # Set Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=data_train,
        eval_dataset=data_valid
    )

    try:
        # Train and save model
        trainer.train(resume_from_checkpoint=pretrain_config['load_checkpoint'])
    except RuntimeError as e:
        print(f"RuntimeError during training: {e}")
        raise

    trainer.save_model(pretrain_config["save_path"])

if __name__ == "__main__":
    pretrain_config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    main(pretrain_config)
