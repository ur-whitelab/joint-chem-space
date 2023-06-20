import sys
sys.path.append("..")

import pandas as pd
import gzip
import json
from time import sleep
from chemspace.pug_utils import get_pug_view_page, regulate_api_requests

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
            self.CIDs = compound_df['CID']
            return
        
    def add_SMILES(self, data_path: str = '../chemspace/Dataset/Data/CID-SMILES.gz'):
        concat_df = pd.DataFrame()
        self.SMILES_df = pd.DataFrame(self.CIDs)
        
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
                break
            # Otherwise, merge the CIDs and SMILES chunk dataframes, and concat. Store as dataset
            elif self._external_CIDs_in_dataset(df['CID']):
                merged_df = self.CIDs.to_frame().merge(df, how='inner', left_on = 'CID', right_on='CID', suffixes= (None,'_y'))
                concat_df = pd.concat([concat_df, merged_df], axis = 0, ignore_index=True)
            
            # Advance counter
            i = i + 1
            self.SMILES_df = concat_df
        self.dataset = self.dataset.merge(self.SMILES_df, how = 'inner', left_on='CID', right_on='CID')
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
        return self.CIDs.isin(external_CIDs).any()

    def add_pubchem_text(self,):
        self.no_CID = 0
        self.text_df = pd.DataFrame(self.CIDs)

        response, pug_view_page_one = get_pug_view_page()
        self._add_pubchem_text(pug_view_page_one)
        wait_time = regulate_api_requests(response)
        total_pages = pug_view_page_one['Annotations']['TotalPages']        
        for page in range(2,total_pages+1):
            sleep(wait_time)
            response, pug_view_page = get_pug_view_page(page=page)
            wait_time = regulate_api_requests(response)
            self._add_pubchem_text(pug_view_page)
            print(f'Page: {page}')

        self.concat_text()

        self.dataset = self.dataset.merge(self.text_df, how = 'inner', left_on = 'CID', right_on= 'CID')

        return
    
    def _add_pubchem_text(self, body: dict):
        #print(body)
        description_list = body['Annotations']['Annotation']
        for description in description_list:
            if 'LinkedRecords' in description.keys() and 'CID' in description['LinkedRecords'].keys():
                CID = description['LinkedRecords']['CID'][0]
                description_source = description['SourceName']
                if 'Description' not in description['Data'][0].keys():
                    #print(description)
                    description_type = 'Undefined'
                else:
                    description_type = description['Data'][0]['Description']
                description_text = description['Data'][0]['Value']['StringWithMarkup'][0]['String']
                #print(description_text)
                col_name = description_type.replace(" ","")
                if col_name not in self.text_df.columns:
                    self.text_df.insert(len(self.text_df.columns),f'{col_name}', [None] * len(self.text_df), allow_duplicates=False)

                # Append description text if 2nd Undefined description
                index = self.text_df.index[(self.text_df.CID == CID)]
                if not index.empty:
                    index = index[0]
                    if col_name == 'Undefined' and (self.text_df.at[index, description_type.replace(" ","")] is not None):
                        self.text_df.loc[index, description_type.replace(" ","")] = self.text_df.at[index, description_type.replace(" ","")] + " " + description_text
                    # Otherwise just assign the value to the correct index
                    else:
                        self.text_df.loc[index, description_type.replace(" ","")] = description_text
                else:
                    continue
            else:
                self.no_CID = self.no_CID + 1
                continue

        return

    def concat_text(self, 
                    cols_to_concat: list = ['OntologySummary', 
                                            'PhysicalDescription', 
                                            'HazardsSummary', 
                                            'LiverToxSummary', 
                                            'Undefined', 
                                            'FDAPharmacologySummary', 
                                            'HIV/AIDSandOpportunisticInfectionDrugs']):
        
        self.text_df['AllText'] = self.text_df.apply( \
            lambda x: '  '.join(filter(None, (x[column] for column in cols_to_concat))), axis=1\
                )
        
        return