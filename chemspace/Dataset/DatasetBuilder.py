import pandas as pd
import gzip
import json

class DatasetBuilder:
    def __init__(
        self,
        compound_file_path: str = None,
        compound_df: pd.DataFrame = None,
        ):
        """
        Instantiate a DatasetBuilder instance, 
        begin the construction of a dataset by creating a dataframe with the CIDs as indices
        
        Args:
            compound_file_path: Optional. Path to a Chemical Structure Records compressed .JSON file downloaded from a PubChem Query
            compund_df: Optional. A dataframe with CIDs as the indices
        """
        # Initialize list to hold CIDs
        CIDs = []

        # If path passed in, open as appropriate and get CIDs
        if compound_file_path:
            # If file is the compressed json file from PubChem, parse and save
            if compound_file_path.endswith('.json.gz'):
                with gzip.open(compound_file_path, 'rt') as zipfile:
                    for line in zipfile:
                        try:
                            dict = json.loads(line[0:-2])                        
                            CIDs.append(dict['id']['id']['cid'])
                        except(json.JSONDecodeError):
                            continue
                        
                    self.CIDs = pd.DataFrame(index = CIDs)
                    self.dataset = pd.DataFrame()
                    return
            # IF passing in a csv, open as appropriate
            elif compound_file_path.endswith('.csv'):
                self.CIDs = pd.read_csv(compound_file_path, index_col='Unnamed: 0')
                self.dataset = pd.DataFrame()
                return
        # If dataframe passed in, assign to self.dataset
        elif compound_df is not None:
            self.dataset = compound_df
            self.CIDs = compound_df.index
            return
        
    def add_SMILES(self, data_path: str = '../chemspace/Dataset/Data/CID-SMILES.gz'):
        data_reader = pd.read_csv(data_path, chunksize= 10 ** 6, sep= "\t", names = ['CID', 'SMILES'], index_col = 'CID')

        for df in data_reader:
            print(df)

        return
    
