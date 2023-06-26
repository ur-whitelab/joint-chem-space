import sys
sys.path.append("..")

import chromadb
import torch
from chemspace import ProjConfig, Projector
from chemspace import Encoder
from torch.utils.data import Dataset
from typing import Optional, Union
import pandas as pd

client = chromadb.Client()
collection = client.create_collection(name="smiles", metadata={"hnsw:space": "l2"})

class PubChemDataset(Dataset):
    def __init__(self,
                 path: str,
                 frac: Optional[Union[float, None]] = None
                ) -> None:
        self.data = pd.read_csv(path)
        self.data = self.data[['SMILES', 'AllText']].dropna().reset_index(drop=True)
        if frac:
            self.data = self.data.sample(frac=frac).reset_index(drop=True)

    def __getitem__(self, index):
        sml = self.data["SMILES"][index]
        desc = self.data["AllText"][index]
        return {
            "sml": sml,
            "desc": desc
        }

    def __len__(self) -> int:
        return len(self.data)


class QueryFromSmiles():
    def __init__(self, dataset, collection, config):
        self.dataset  = dataset
        self.collection = collection
        self.config = config
        self.E_desc = Encoder()
        self.P_desc = Projector(**vars(config))
        self.P_desc.load_state_dict(torch.load("../experiments/P_desc.pt"))

    def query(self, desc, n_query=5):
        desc_emd = self.P_desc(self.E_desc(desc))
        out = self.collection.query(
            query_embeddings=desc_emd.flatten().tolist(),
            n_results=n_query
        )
        return out
        

def create_chroma_collection(dataset, config):
    client = chromadb.Client()
    collection = client.create_collection(name="smiles", metadata={"hnsw:space": "l2"})
    P_sml_config = config
    P_sml = Projector(**vars(P_sml_config))
    E_sml = Encoder()
    P_sml.load_state_dict(torch.load("../experiments/P_sml.pt"))
    ids = []
    descs = []
    embeddings = []
    metadatas = []
    documents = []
    for id in range(100):
        ids.append(f"{id}")
        record = chemdata.__getitem__(id)
        smile = record['sml']
        documents.append(smile)
        smile_emd = P_sml(E_sml(smile))
        embeddings.append(smile_emd.flatten().tolist())
        #metadatas.append({f"m{id}": {id}})

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        #metadatas=metadatas
    )
    return collection

def run(desc):
    chemdata = PubChemDataset('../Dataset.csv.gz')
    config = ProjConfig(
        input_size=384,
        hidden_size=256,
        output_size=512,
    )
    collection = create_chroma_collection(chemdata, config)
    query_from_smiles = QueryFromSmiles(chemdata, collection, config)
    out = query_from_smiles.query(desc)
    print(out)

chemdata = PubChemDataset('../Dataset.csv.gz')
run(chemdata.__getitem__(10)['desc'])
