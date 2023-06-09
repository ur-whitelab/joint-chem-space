import sys
sys.path.append("..")

import pandas as pd
import gzip
import json
import numpy as np
from time import sleep

from rdkit.Chem import MolFromSmiles

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
            elif compound_file_path.endswith('.csv') or compound_file_path.endswith('.csv.gz'):
                self.dataset = pd.read_csv(compound_file_path)
                self.CIDs = self.dataset['CID']
                return
        # If dataframe passed in, assign to self.dataset
        elif compound_df is not None:
            self.dataset = compound_df
            self.CIDs = compound_df['CID']
            return
        
    def add_SMILES(self, data_path: str = '../chemspace/Dataset/Data/CID-SMILES.gz') -> None:
        """
        Method to add SMILES information from a zipped file from the PubChem ftp server to the dataset
        Args:
            datapath: path to the zipped csv file from PubChem contianing SMILES

        """
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
        self.count_atoms_in_compunds()
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

    def count_atoms_in_compunds(self):
        """
        Method to add a column to the dataset containing the number of atoms in each compound
        """
        self.dataset['NumAtoms'] = self.dataset['SMILES'].apply(lambda x: self._get_no_atoms(x))

        return

    def _get_no_atoms(self, SMILES):
        """
        Get the number of atoms in a compound from its SMILES representation
        Args:
            SMILES: string containing SMILES representation for one compound
        Returns:
            int: number of atoms in the compound
            None: if invalid SMILES that can't be converted by rdkit
        """
        m = MolFromSmiles(SMILES)
        if not m:
            return None
        else:
            return m.GetNumAtoms()

    def add_pubchem_text(self) -> None:
        """
        Method to add text from PUG View api to dataset
        Sends a request to the PUG View api for a page, and iterates through all pages to get all descriptions
        """
        # Initialize counter for records without CIDs linked and dataframe to hold all text descriptions
        self.no_CID = 0
        self.text_df = pd.DataFrame(self.CIDs)

        # Get first page and determine appropriate wait time before next request
        response, pug_view_page_one = get_pug_view_page()
        self._add_pubchem_text(pug_view_page_one)
        wait_time = regulate_api_requests(response)
        
        # Get total number of pages avaliable in PUG View from first page info
        total_pages = pug_view_page_one['Annotations']['TotalPages']        

        # Iterate from 2nd page through all pages avaliable
        for page in range(2,total_pages+1):
            # Wait as appropriate before sending next request
            sleep(wait_time)
            # Send next request and store response and response body
            response, pug_view_page = get_pug_view_page(page=page)
            # Calculate next wait time
            wait_time = regulate_api_requests(response)
            # Send page to method to parse page and add descriptions to text_df dataframe
            self._add_pubchem_text(pug_view_page)
            # Display page number completed
            print(f'Page: {page}/{total_pages}')

        # Concatenate all of the text descriptions for each compound into a single column
        self.concat_text()

        # Merge the dataframe containing text descriptions with the dataset
        self.dataset = self.dataset.merge(self.text_df, how = 'inner', left_on = 'CID', right_on= 'CID')

        return
    
    def _add_pubchem_text(self, body: dict) -> None:
        """
        Method to add the body of a single PUG View page to a dataframe contianing only the textual descriptions from pubchem
        Args:
            body: response body of the api request, as a dictionary
        TODO: Consider if we want to change how we handle cases where there is no CID. 
                After name and synonym information is added we may want to add anyway, matching on name, 
                and see if there are SMILES associated with it since we're training on SMILES and text
        """
        # Parse the dictionary to get the list of all descriptions for the compounds in the page
        description_list = body['Annotations']['Annotation']

        # Iterate through list of descriptions
        for description in description_list:
            
            # Check the description for a CID in 'LinkedRecords' 
            # If one is present, we want to add the information to the dataset
            if 'LinkedRecords' in description.keys() and 'CID' in description['LinkedRecords'].keys():
                
                # Get information of interest from the compound description dictionary
                # interested in CID, description type (used as column names for organization), 
                # and text description itself. 
                # In the future we may want to use description source, so we're gathering it here just in case
                CID = description['LinkedRecords']['CID'][0]
                description_source = description['SourceName']
                description_text = description['Data'][0]['Value']['StringWithMarkup'][0]['String']
                
                # If there is no information given for what kind of description it is, mark as undefined so we can still save
                if 'Description' not in description['Data'][0].keys():
                    description_type = 'Undefined'
                else:
                    description_type = description['Data'][0]['Description']
                
                # Create appropriate column name from description type
                col_name = description_type.replace(" ","")
                # If a column with that name doesn't already exist in the dataframe, 
                # create a new empty column and add
                if col_name not in self.text_df.columns:
                    self.text_df.insert(len(self.text_df.columns),f'{col_name}', [None] * len(self.text_df), allow_duplicates=False)

                # Get the index of the dataframe of the compound with the matching CID
                # `index` is type pd.DataFrame here
                index = self.text_df.index[(self.text_df.CID == CID)]
                
                # If there is an index, add the description to the dataframe as appropriate
                if not index.empty:
                    # Get the index from the DataFrame
                    index = index[0]
                    # Append description text if 2nd Undefined description for the compound. 
                    # This prevents overwriting the first undefined description when more than one is present
                    if col_name == 'Undefined' and (self.text_df.at[index, description_type.replace(" ","")] is not None):
                        self.text_df.loc[index, description_type.replace(" ","")] = self.text_df.at[index, description_type.replace(" ","")] + " " + description_text
                    # Otherwise just assign the value to the correct index
                    else:
                        self.text_df.loc[index, description_type.replace(" ","")] = description_text
                # If an index is not present, we cannot add to dataset
                else:
                    continue
            else:
                # If there is no CID information or linked record, increase counter for missing CIDs
                self.no_CID = self.no_CID + 1
                continue

        return

    def add_s2r_text(self, s2r_path: str = "../chemspace/Dataset/Data/out.csv"):
        # TODO: change aggregator function to join desciptions 
        # with a different character to avoid interferring with the molecule name masks
        
        chunksize = 10 ** 6
        s2r_reader = pd.read_csv(s2r_path, chunksize = chunksize, names=['Name', 'CID', 'Description', 'PaperID'], usecols=['CID', 'Description'],index_col=None)

        concat_df = pd.DataFrame()

        uniques = np.array([], dtype=np.int64)

        for i, df in enumerate(s2r_reader):
            print(f"Chunk {i}", end='\r')

            df = df.groupby('CID')
            df = df.agg('|'.join).reset_index()

            concat_df = pd.concat([concat_df, df], ignore_index=True, axis=0)

            
            

        concat_df = concat_df.groupby('CID')
        concat_df = concat_df.agg('|'.join).reset_index()

        print('Saving Data')
        concat_df.to_csv('../chemspace/Dataset/Data/s2rtext.csv', index=False)

        return

    def concat_text(self, 
                    cols_to_concat: list = ['OntologySummary', 
                                            'PhysicalDescription', 
                                            'HazardsSummary', 
                                            'LiverToxSummary', 
                                            'Undefined', 
                                            'FDAPharmacologySummary', 
                                            'HIV/AIDSandOpportunisticInfectionDrugs']) -> None:
        """
        Method to concatenate the values of textual columns 
        into one column that contains all text descriptions for a given compound in each row
        Args:
            cols_to_concate: list of columns to concatenate.
            `                   Can be adjusted as more types of text descriptions become of interest
        """
        
        # Concatenate the values for all the text description columns (on a row by row basis) 
        # into one value for a new column called `AllText`
        self.text_df['AllText'] = self.text_df.apply( \
            lambda x: '  '.join(filter(None, (x[column] for column in cols_to_concat))), axis=1\
                )
        
        # Replace any `''` values (generated when all descriptions are None) with a None value
        self.text_df.replace(to_replace = '', value = None, inplace=True)
        
        return
    
    def clean_dataset(self) -> None:
        """
        Method to perform cleaning operations to the dataset
        As specific methods are added they can be called here so that they can all be run easily
        """

        df_lenth = len(self.dataset)

        if 'AllText' in self.dataset.columns:
        # Remove any rows for compounds that have no descriptions
            self._remove_empty_rows_in_column(column='AllText')
            rows_removed = df_lenth - len(self.dataset)
            print(f"{rows_removed} compunds with no descriptions removed from dataset")
            df_lenth = len(self.dataset)

        if 'NumAtoms' in self.dataset.columns:
            # Remove any rows for compounds that have invalid SMILES
            self._remove_empty_rows_in_column(column='NumAtoms')
            rows_removed = df_lenth - len(self.dataset)
            print(f"{rows_removed} compunds with invalid SMILES removed from dataset")

            df_lenth = len(self.dataset)

        return

    def _remove_empty_rows_in_column(self, column: str = None) -> None:
        """
        Method to remove rows that have no description at all from the dataset
        """
        # Remove rows where the `AllText` value is None 
        # so that training is not negatively impacted by mixed type columns
        self.dataset.dropna(subset=[column], inplace=True, ignore_index = True)
        return

    def _get_synonyms_file(self) -> None:
        raise NotImplementedError

    def _process_synonyms_file(self, file_path: str, max_CID: int) -> pd.DataFrame:
        """
        Method to process the CID-Synonym-filtered.gz file.
        This file should be downloaded from PubChem and is not provided in the repository
        """
        # Create a dictionary to hold the contents of the file
        content = {
            "CID": [],
            "Synonyms": []
        }
        # Open the file and read the contents into the dictionary
        with gzip.open(file_path, 'rt') as zipfile:
            for i, line in enumerate(zipfile):
                CID, synonym = line.split()[0], " ".join(line.split()[1:])
                content['CID'].append(int(CID))
                content['Synonyms'].append(synonym)
                if (max_CID is not None) and (i > max_CID):
                    break
        # Convert the dictionary to a dataframe
        return pd.DataFrame.from_dict(content)

    def add_synonyms(self, file_path: str = '../chemspace/Dataset/Data/CID-Synonym-filtered.gz', max_CID: int = None) -> None:
        """
        Method to add synonyms to the dataset
        """

        synonyms_df = self._process_synonyms_file(file_path, max_CID)
        synonyms_df = synonyms_df.groupby('CID')['Synonyms'].apply('; '.join).reset_index()
        synonyms_df['Number_of_Synonyms'] = synonyms_df['Synonyms'].apply(lambda x: len(x.split('; ')))

        # Merge the synonyms dataframe with the dataset
        self.dataset = self.dataset.merge(synonyms_df, how = 'outer', on='CID')

        return
