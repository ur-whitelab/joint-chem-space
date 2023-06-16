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
                        
                    self.CIDs = pd.Series(CIDs, name='CID', dtype = 'int64')
                    self.dataset = pd.DataFrame()
                    return
            # IF passing in a csv, open as appropriate
            elif compound_file_path.endswith('.csv'):
                self.dataset = pd.read_csv(compound_file_path,)
                self.CIDs = self.dataset['CID']
                return
        # If dataframe passed in, assign to self.dataset
        elif compound_df is not None:
            self.dataset = compound_df
            self.CIDs = compound_df.index
            return
        
    def add_SMILES(self, data_path: str = '../chemspace/Dataset/Data/CID-SMILES.gz'):
        concat_df = pd.DataFrame()
        
        # create object to iterate through CSV chunks
        data_reader = pd.read_csv(data_path, chunksize= 10 ** 6, sep= "\t", names = ['CID', 'SMILES'])
        
        # Initialize counter and 
        i = 0

        # Iterate through df chunks
        for df in data_reader:
            # Display progress
            if i % 5 == 0:
                print(i)

            # if the lowest CID of the chunk is greater than the largest CID we're interested in, then stop
            if df['CID'].iloc[0] > self.CIDs.iloc[-1]:
                return
            # Otherwise, merge the CIDs and SMILES chunk dataframes, and concat. Store as dataset
            elif self._external_CIDs_in_dataset(df['CID']):
                merged_df = self.CIDs.to_frame().merge(df, how='inner', left_on = 'CID', right_on='CID', suffixes= (None,'_y'))
                concat_df = pd.concat([concat_df, merged_df], axis = 0, ignore_index=True)
            
            # Advance counter
            i = i + 1
            self.dataset = concat_df
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
