import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, 
                          AutoTokenizer, 
                          AutoModelForSeq2SeqLM,
                          AutoModel,
                          AutoModelForMaskedLM)
from typing import Any

# model_list = [
#     AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MTR"),
#     AutoModelForSeq2SeqLM.from_pretrained("laituan245/molt5-small-smiles2caption"),
#     AutoModel.from_pretrained('allenai/scibert_scivocab_cased'),
# ]

class SML_Encoder:
    def __init__(self,
                 model_name: str = "DeepChem/ChemBERTa-77M-MLM") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def tokenize(self, x: str) -> torch.Tensor:
        return self.tokenizer(x, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

    def __call__(self, tokens: str) -> torch.Tensor:
        return self.model(**tokens, output_hidden_states=True)[1][-1]
    

class TXT_Encoder:
    def __init__(self,
                 model_name: str = "allenai/scibert_scivocab_cased") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def tokenize(self, x: str) -> torch.Tensor:
        return self.tokenizer(x, return_tensors="pt", padding='max_length', truncation=True, max_length=512)

    def __call__(self, tokens: str) -> torch.Tensor:
        return self.model(**tokens, output_hidden_states=True)[1][-1]


if __name__ == "__main__":
    smiles = SML_Encoder()
    txt = TXT_Encoder()
    test_sml = smiles.tokenize["CCO"]
    test_txt = txt.tokenize(["To synthesize CCO, we need to do this and that"])
    print(txt(test_txt).shape)