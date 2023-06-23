import sys
sys.path.append(".")

import pandas as pd
import os
import pytest

import chemspace as cs
from chemspace.Dataset.DatasetBuilder import DatasetBuilder
from chemspace.pug_utils import get_pug_view_page

@pytest.fixture
def pubchem_compund_report_path():
    return os.path.abspath('./chemspace/Dataset/Data/PubChem_compound_list_records.json.gz')

@pytest.fixture
def dataset_CSV_path():
    return os.path.abspath('./chemspace/Dataset/Data/CIDs.csv')

@pytest.fixture
def CID_df(dataset_CSV_path):
    return pd.read_csv(dataset_CSV_path)

@pytest.fixture
def pug_view_page_one():
    response, pug_view_page_one = get_pug_view_page()
    return pug_view_page_one

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
        assert not DB.CIDs.empty
        # Check number of CIDs imported
        assert len(DB.CIDs) > 250,000

    def test_instantiate_DB_from_DF(self, CID_df):
        """
        Tests to cover instantiating a DatasetBuilder object from a dataframe
        """
        # Create Dataset Builder instance
        DB = DatasetBuilder(compound_df=CID_df)
        
        # Check that indexed dataframe was used for the instance dataset
        assert not DB.CIDs.empty
        # Check Number of CIDs imported
        assert len(DB.CIDs) > 250,000

    def test_add_pubchem_text(self, CID_df, pug_view_page_one):
        """
        Test to cover method for adding information from a PUG View page to a dataset
        """
        # Create Dataset Builder instance
        DB = DatasetBuilder(compound_df=CID_df)
        DB.text_df = pd.DataFrame(DB.CIDs)
        DB.no_CID = 0
        DB._add_pubchem_text(pug_view_page_one)

        # Assert that columns were added to the text dataframe
        assert len(DB.text_df.columns) > 1

        # assert that each column has non Null values
        assert (DB.text_df['HazardsSummary'].notna()).any()
        assert (DB.text_df['PhysicalDescription'].notna()).any()

    def test_concatenate_columns(self, CID_df, pug_view_page_one):
        """
        Unit test for DatasetBuilder.concat_text()
        """
        # Create Dataset Builder instance
        DB = DatasetBuilder(compound_df=CID_df)
        DB.text_df = pd.DataFrame(DB.CIDs)
        DB.no_CID = 0
        DB._add_pubchem_text(pug_view_page_one)

        # Concatenate text descriptions
        DB.concat_text(cols_to_concat=DB.text_df.columns.drop('CID'))

        # Ensure new column was created and that there are non-null values present
        assert 'AllText' in DB.text_df.columns
        assert (DB.text_df['AllText'].notna()).any()

    def test_clean_dataset(self, CID_df, pug_view_page_one):
        """
        Unit test for DatasetBuilder.clean_dataset()
        """
        # Create Dataset Builder instance
        DB = DatasetBuilder(compound_df=CID_df)
        DB.text_df = pd.DataFrame(DB.CIDs)
        DB.no_CID = 0
        DB._add_pubchem_text(pug_view_page_one)
        DB.add_SMILES(data_path='./chemspace/Dataset/Data/CID-SMILES.gz')

        # Concatenate text
        DB.concat_text(cols_to_concat=DB.text_df.columns.drop('CID'))

        # Merge dataframes to update dataset value
        DB.dataset = DB.dataset.merge(DB.text_df, how = 'inner', left_on = 'CID', right_on= 'CID')

        # Measure number of rows in dataset before dropping rows
        orginal_length = len(DB.dataset)
        # Drop rows with Null values for `AllText`
        DB.clean_dataset()

        # Assert only non-null values are left and that there are less rows than orignially
        assert (DB.dataset['AllText'].notna()).all()
        assert len(DB.dataset) < orginal_length
        