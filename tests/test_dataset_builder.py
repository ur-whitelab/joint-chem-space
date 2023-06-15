import sys
sys.path.append(".")

import pandas as pd
import os
import pytest

import chemspace as cs
from chemspace.Dataset.DatasetBuilder import DatasetBuilder


@pytest.fixture
def pubchem_compund_report_path():
    return os.path.abspath('./chemspace/Dataset/Data/PubChem_compound_list_records.json.gz')

@pytest.fixture
def dataset_CSV_path():
    return os.path.abspath('./chemspace/Dataset/Data/CIDs.csv')

@pytest.fixture
def CID_df(dataset_CSV_path):
    return pd.read_csv(dataset_CSV_path, index_col='Unnamed: 0')

class TestDatasetBuilder:
    
    @pytest.mark.zipped_files
    @pytest.mark.parametrize('compound_file_path',['pubchem_compund_report_path', 'dataset_CSV_path'])
    def test_instantiate_DB_from_file(self, compound_file_path, request):
        """
        Tests to cover instantiating a DatasetBuilder object from a zipped JSON file or a CSV
        """
        # Get the path to the file and create DatasetBuilder instance
        compound_file_path = request.getfixturevalue(compound_file_path)
        DB = DatasetBuilder(compound_file_path=compound_file_path)

        # Check that indexed dataframe was created
        assert not DB.dataset.index.empty

    def test_instantiate_DB_from_DF(self, CID_df):
        """
        Tests to cover instantiating a DatasetBuilder object from a dataframe
        """
        # Create Dataset Builder instance
        DB = DatasetBuilder(compound_df=CID_df)
        
        # Check that indexed dataframe was used for the instance dataset
        assert not DB.dataset.index.empty

