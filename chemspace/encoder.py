import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback, TrainingArguments, Trainer

#create data handlers
#torchy way of creating dataclass
class data_loader:
    def __init__(self, i_data):
        self.data = i_data

    def get_split(self, train_ratio=0.75, valid_ratio=0.15, seed=None):
        n = len(self.data)
        indices = np.arange(n)
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
        train_size = int(train_ratio * n)
        valid_size = int(valid_ratio * n)
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size+valid_size]
        test_indices = indices[train_size+valid_size:]
        i_train_data = self.data.iloc[train_indices].reset_index(drop=True)
        i_valid_data = self.data.iloc[valid_indices].reset_index(drop=True)
        i_test_data = self.data.iloc[test_indices].reset_index(drop=True)
        return i_train_data, i_valid_data, i_test_data
    
    class Input(Dataset):
    def __init__(self, i_data, i_tokenizer, i_max_length):
        self.data = i_data
        self.tokenizer = i_tokenizer
        self.max_length = i_max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]["SMILES"]
        inputs = self.tokenizer(smiles, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_length)
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"].squeeze(0)
        inputs["labels"] = torch.tensor(self.data.iloc[idx]["Col"], dtype=torch.float).unsqueeze(0)
        return inputs
    

# encoder 

#AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Load a pretrained transformer model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
config.num_hidden_layers += 1
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)


#see if GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")
# move model to the device
model.to(device)