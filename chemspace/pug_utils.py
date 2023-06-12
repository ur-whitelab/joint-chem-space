import requests
import time
from re import search
from typing import Dict

import pandas as pd

# def get_compound(cid):
#     '''
#     This code uses the pubchem API to retrieve a compound based on its cid (compound ID). 
#     The cid is used to make a request to the pubchem API and retrieve the compound information in JSON format.
#     '''    
#     base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
#     compound_url = f"{base_url}/compound/cid/{cid}/smiles/JSON"
#     response = requests.get(compound_url)
    
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return None


# def get_compound_by_name(compound_name):
#     '''
#     This code takes a compound name from the user and
#     retrieves the compound from PubChem.
#     '''
#     base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
#     compound_url = f"{base_url}/compound/name/{compound_name}/JSON"
#     response = requests.get(compound_url)

#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Failed to get compound: {compound_name}")
#         return None


def get_compound_name_and_smiles(cid):
    '''
    This function takes a compound id and returns the IUPAC name and SMILES string for that compound
    '''
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    compound_url = f"{base_url}/compound/cid/{cid}/property/IUPACName,CanonicalSMILES/JSON"
    response = requests.get(compound_url)

    if response.status_code == 200:
        data = response.json()
        try:
            properties = data["PropertyTable"]["Properties"]
            return response, properties[0]["IUPACName"], properties[0]["CanonicalSMILES"]
        except:
          print(f'error download data for cid {cid}')
    return response, None, None


def get_compound_description(cid):
    """
    Get compound description from PubChem
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    description_url = f"{base_url}/compound/cid/{cid}/description/JSON"
    response = requests.get(description_url)

    if response.status_code == 200:
        data = response.json()
        try:
            return response, data['InformationList']["Information"][1]['Description']
        except IndexError:
            return response, "No description available."
    else:
        print(f"Failed to get description for compound CID: {cid}")
        return response, "-"
    

def download_compounds(start_cid, end_cid):
    """
    Downloads compounds between the start_cid and end_cid, inclusive.
    Returns the names, smiles, and descriptions of the compounds.
    """
    names = []
    smiless = []
    descriptions = []
    for cid in range(start_cid, end_cid+1):
        # Send request to get compound name and SMILES
        name_response, c_name, c_smiles = get_compound_name_and_smiles(cid)
        
        # Determine appropriate wait time before sending next request and wait
        wait_time = regulate_api_requests(name_response)
        time.sleep(wait_time)
        
        # Send request to get compound description
        desc_response, desc = get_compound_description(cid)
        # Determine appropriate wait time before sending next request
        wait_time = regulate_api_requests(desc_response)
        
        if c_name is not None:
            names.append(c_name)
            smiless.append(c_smiles)
            descriptions.append(desc)
            print(f"Downloaded compound {cid}")
        else:
            print(f"Failed to download compound {cid}")
        
        # Wait before continuing loop and sending next request
        time.sleep(wait_time)
        
    return [name_response, desc_response], names, smiless, descriptions


def regulate_api_requests(response: str) -> float:
    wait_time = 0.2
    
    statuses = parse_throttling_headers(response.headers['X-Throttling-Control'])

    if (statuses['status'] == 'green').all():
        return wait_time
    if (statuses['status'] == 'black').any():
        wait_time = 3600
    elif (statuses['status'] == 'red').any():
        wait_time = 300
    elif (statuses['status'] == 'yellow').any():
        wait_time = 60

    return wait_time


def parse_throttling_headers(throttle_str: str) -> pd.DataFrame:
    """
    Function to parse the API throttling headers into a usable dictionary
    Args:
        throttle_str: String of the api response headers related to API throttling Status
    Returns:
        statuses: Dataframe containing information from the headers 
        in the form of {Status_Type: {'status': <status color>, 'percent_load': <percentage>},...}
    """
    # Initialize status dictionary
    status_dict = {}

    # Split the string of the statuses into parts relating to each status type
    statuses = throttle_str.split(", ")

    # Split out keys and value sets
    keys = [status_type.split(' status')[0] for status_type in statuses]
    value_sets = [status_info.split(' status')[1][2:] for status_info in statuses]

    # build nested dictionary of status information
    for key, value_set in zip(keys,value_sets):
        status_dict[key] = {
            'status': search('[a-zA-Z]*', value_set)[0].lower(),
            'percent': int(search('\d{1,3}', value_set)[0]),
        }

    statuses = pd.DataFrame(status_dict).T

    return statuses
