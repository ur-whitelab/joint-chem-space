import pandas as pd
import gzip
import json
import pandas as pd
from json import JSONDecodeError

class DatasetBuilder:
    def __init__(
        self,
        compound_list_path: str = None,
        compound_df: pd.DataFrame = None,
        ):
        # Initialize list to hold CIDs
        self.CIDs = []

        # If path passed in, open and parse lines to get CIDs
        if compound_list_path:
            with gzip.open(compound_list_path, 'rt') as zipfile:
                for line in zipfile:
                    data = zipfile.readline()
                    if data and len(data) > 3:
                        dict = json.loads(data[0:-2])                        
                        self.CIDs.append(dict['id']['id']['cid'])
                    else:
                        self.dataset = pd.DataFrame(index = self.CIDs)
                        return
        # If dataframe passed in, assign to self.dataset
        elif compound_df:
            self.dataset = compound_df
            return
        