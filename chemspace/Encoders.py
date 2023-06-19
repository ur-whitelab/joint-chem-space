import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (AutoConfig, 
                          AutoTokenizer, 
                          AutoModelForSeq2SeqLM,
                          AutoModel,
                          AutoModelForMaskedLM)
from typing import Any
from mace.modules.models import MACE
import e3nn
from mace import modules as MACE_modules
#from e3nn.util import jit
from mace import data, tools


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


class Encoder3D:
    def __init__(self, out_dim, num_elements):
        self.out_dim = out_dim
        self.config = dict(
            r_max=5,
            num_bessel=8,
            num_polynomial_cutoff=6,
            max_ell=2,
            interaction_cls=MACE_modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=MACE_modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=5,
            num_elements=num_elements,
            hidden_irreps=e3nn.o3.Irreps("32x0e + 32x1o"),
            MLP_irreps=e3nn.o3.Irreps("16x0e"),
            gate=torch.nn.functional.silu,
            avg_num_neighbors=8,
            atomic_numbers=16,
            atomic_energies=np.array([float(k) for k in range(num_elements)]),
            correlation=3,
        )
        self.model = MACE_modules.MACE(**self.config)
        

    def __call__(self, xyz_path):
        _, data_config = data.load_from_xyz(xyz_path, config_type_weights={"Default": 1.0})
        table = tools.AtomicNumberTable([k for k in range(1,101)])
        atomic_data = []
        for config in data_config:
            #self.model = MACE_modules.MACE(**self.config)
            atomic_data.append(data.AtomicData.from_config(config, z_table=table, cutoff=3.0))
        data_loader = tools.torch_geometric.dataloader.DataLoader(
            dataset=atomic_data,
            batch_size=32,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader))
        return self.model(batch.to_dict(), training=True)
        
if __name__ == "__main__":
    #m = Encoder()
    #test = ["CCO"]
    #print(m(test).shape)
    
    xyz_path = './zy.xyz'
    #xyz_path = '../experiments/mol.xyz'
    m3d = Encoder3D(1, 100)
    print(m3d(xyz_path)['node_feat'])
