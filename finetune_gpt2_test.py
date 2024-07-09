# Preprocess Data: TransPolymer uses a tokenizer for processing polymer SMILES strings

# defines a custom dataset class for preprocessing polymer SMILES data for training ML models

    # imports
from PolymerSmilesTokenization import PolymerSmilesTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
    
    # custom dataset class:
##class LoadPretrainData(Dataset):
def __init__(self, tokenizer, dataset, blocksize): # blocksize = max length for tokenization
    self.tokenizer = tokenizer
    self.blocksize = blocksize
    self.dataset = dataset

def __len__(self): # returns length of the dataset
    return len(self.dataset)

def __getitem__(self, i): # takes an index 'i' and retrives the SMILES string from the dataset
    smiles = self.dataset[i][0]
    encoding = self.tokenizer(
        str(smiles),
        add_special_tokens=True,     # adds special tokens
        max_length=self.blocksize,   # sets max len of sequence
        padding="max_length",
        truncation=True,             
        return_attention_mask=True,
        return_tensors='pt',         # returns tokenized output as pytorch tensors
    )
    return {
        "input_ids": encoding["input_ids"].flatten(),
        "attention_mask": encoding["attention_mask"].flatten(),
    }

if __name__ == "__main__":
    print(1)
    # Example usage:
    tokenizer = PolymerSmilesTokenizer()
    print("PolymerSmilesTokenizer instance created")

    data = [("C(C(=O)O)N",), ("CC(C)C(=O)O",)]  # Example SMILES dataset
    print("Example SMILES dataset created")

    blocksize = 128
    dataset = LoadPretrainData(tokenizer, data, blocksize)
    print("LoadPretrainData instance created")
    print(1)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print("DataLoader instance created")

    for batch in dataloader:
        print("Batch received")
        print(batch["input_ids"])
        print(batch["attention_mask"])
