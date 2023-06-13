import pandas as pd
import gzip
import json
import pandas as pd
from json import JSONDecodeError

from typing import List

class DatasetBuilder:
    def __init__(
        self,
        compound_list_path: str = None,
        compound_list: List = None,
        ):
        self.CIDs = []
        if compound_list_path:
            with gzip.open(compound_list_path, 'rt') as zipfile:
                for line in zipfile:
                    data = zipfile.readline()
                    if data and len(data) > 3:
                        dict = json.loads(data[0:-2])                        
                        self.CIDs.append(dict['id']['id']['cid'])
                    else:
                        break

        self.dataset = pd.DataFrame(index = self.CIDs)
        return