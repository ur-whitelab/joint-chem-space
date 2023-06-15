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
        i = 0
        for df in data_reader:
            if i % 5 == 0:
                print(i)

            #print(df)
            if df.index[0] > self.CIDs.index[-1]:
                return
            elif self._external_CIDs_in_dataset(df.index):
                try:
                    self.dataset = pd.concat([self.dataset, self.CIDs.merge(df, how='left', left_index = True, right_index=True, suffixes= (None,'_y'))], axis = 0)
                except(pd.errors.MergeError):
                    print(df)
            i = i + 1
        return
    
    def _external_CIDs_in_dataset(self, external_CIDs: pd.Index) -> bool:
        """
        Function that compares indices of the Dataset to a set of indices passed in.
        Args:
            external_CIDs: CID indices of an external dataframe
        Returns:
            True: If any of the dataset CIDs are present in the external dataframe
            False: If none of the dataset CIDS are present in the external dataframe

        """
        return self.CIDs.index.isin(external_CIDs).any()
