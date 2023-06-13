import pandas as pd
import gzip
import json
import pandas as pd
from json import JSONDecodeError

class DatasetBuilder:
    def __init__(
        self,
        compound_file_path: str = None,
        compound_df: pd.DataFrame = None,
        ):
        # Initialize list to hold CIDs
        self.CIDs = []

        # If path passed in, open as appropriate and get CIDs
        if compound_file_path:
            # If file is the compressed json file from PubChem, parse and save
            if compound_file_path.endswith('.json.gz'):
                with gzip.open(compound_file_path, 'rt') as zipfile:
                    for line in zipfile:
                        data = zipfile.readline()
                        if data and len(data) > 3:
                            dict = json.loads(data[0:-2])                        
                            self.CIDs.append(dict['id']['id']['cid'])
                        else:
                            self.dataset = pd.DataFrame(index = self.CIDs)
                            return
            # IF passing in a csv, open as appropriate
            elif compound_file_path.endswith('.csv'):
                self.dataset = pd.read_csv(compound_file_path, index_col='Unnamed: 0')
                return
        # If dataframe passed in, assign to self.dataset
        elif compound_df:
            self.dataset = compound_df
            return
        