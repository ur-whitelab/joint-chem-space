import sys
sys.path.append("..")

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
    return os.path.abspath('./chemspace/Dataset/Data/CompoundDataset.csv')

class TestDatasetBuilder:
    
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
