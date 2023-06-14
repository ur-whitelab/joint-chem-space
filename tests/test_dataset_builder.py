import sys
sys.path.append("..")

import pandas as pd
import os
import pytest

import chemspace as cs
from chemspace.Dataset.DatasetBuilder import DatasetBuilder

class TestDatasetBuilder:
    
    @pytest.mark.parametrize('compound_file_path',['./chemspace/Dataset/Data/PubChem_compound_list_records.json.gz','./chemspace/Dataset/Data/CompoundDataset.csv'])
    def test_instantiate_DB_from_file(self, compound_file_path):
        """
        Tests to cover instantiating a DatasetBuilder object from a zipped JSON file or a CSV
        """
        # Get the absolute path to the file and create DatasetBuilder instance
        compound_file_path = os.path.abspath(compound_file_path)
        DB = DatasetBuilder(compound_file_path=compound_file_path)

        # Check that indexed dataframe was created
        assert not DB.dataset.index.empty
