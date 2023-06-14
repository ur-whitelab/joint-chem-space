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

class Encoder:
    def __init__(self,
                 model_name: str = "DeepChem/ChemBERTa-77M-MLM") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained("DeepChem/ChemBERTa-77M-MLM")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def __call__(self, x: str) -> torch.Tensor:
        tokens = self.tokenizer(x, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        return self.model(**tokens, output_hidden_states=True)[1][-1]


if __name__ == "__main__":
    m = Encoder()
    test = ["CCO"]
    print(m(test).shape)