import requests
import time
from typing import Tuple, List, Dict
from re import search
import pandas as pd
import json


def get_compound_name_and_smiles(
        cid: int
    ) -> Tuple[str, str]:
    '''
    This function takes a PubChem compound ID and returns the IUPAC name and 
    SMILES string for that compound.

    Args:
        cid (int): PubChem compound ID

    Returns:
        tuple: Contains the IUPAC name and SMILES string of the compound. 
               If the compound cannot be found, return (None, None).
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
          print(f'Error in downloading data for CID: {cid}')
    return response, None, None


def get_compound_description(
        cid: int
    ) -> str:
    """
    Get compound description from PubChem.

    Args:
        cid (int): PubChem compound ID

    Returns:
        str: A description of the compound. If the compound doesn't have a 
             description or cannot be found, it returns appropriate messages.
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


def download_compounds(
        start_cid: int,
        end_cid: int
    ) -> Tuple[List[str], List[str], List[str]]:
    """
    Downloads compound data between the start_cid and end_cid, inclusive.
    Returns the names, SMILES strings, and descriptions of the compounds.

    Args:
        start_cid (int): The starting compound ID in the range.
        end_cid (int): The ending compound ID in the range.

    Returns:
        tuple: Contains three lists:
            names (List[str]): A list of compound names.
            smiless (List[str]): A list of SMILES strings of the compounds.
            descriptions (List[str]): A list of descriptions of the compounds.
    """
    names = []
    smiless = []
    descriptions = []
    wait_time = 0.2
    for cid in range(start_cid, end_cid+1):
        # Send request to get compound name and SMILES
        name_response, c_name, c_smiles = get_compound_name_and_smiles(cid)
        
        # Determine appropriate wait time before sending next request and wait
        wait_time = regulate_api_requests(name_response)

        # Redo last request if it was blocked
        if wait_time >=3600.0:
            time.sleep(wait_time)
            while wait_time >= 3600.0:
               name_response, c_name, c_smiles = get_compound_name_and_smiles(cid) 
               wait_time = regulate_api_requests(name_response)
               time.sleep(wait_time)
        else:
            time.sleep(wait_time)
        
        
        # Send request to get compound description
        desc_response, desc = get_compound_description(cid)
        # Determine appropriate wait time before sending next request
        wait_time = regulate_api_requests(desc_response)
        
        # Redo last request if it was blocked
        if wait_time >=3600.0:
            time.sleep(wait_time)
            while wait_time >= 3600.0:
               desc_response, desc = get_compound_description(cid)
               wait_time = regulate_api_requests(desc_response)
               time.sleep(wait_time)
        else:
            time.sleep(wait_time)

        if c_name is not None:
            names.append(c_name)
            smiless.append(c_smiles)
            descriptions.append(desc)
            print(f"Downloaded compound {cid}")
        else:
            print(f"Failed to download compound {cid}")
        
        # Wait before continuing loop and sending next request
        time.sleep(wait_time)
        
    return names, smiless, descriptions

def get_pug_view_page(heading: str = 'Record Description', page: int = 1):
    """
    Function to send a request to get a page from PUG View
    Args:
        heading: PubChem compund heading of interest
        page: page of PUG View to return
    Returns:
        response: API response
        body: response body packaged as a dictionary        
    """
    # Build URL
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/annotations/heading/JSON"
    heading = heading.replace(' ', '+')
    heading_url = f"heading_type=Compound&heading={heading}&page={page}"
    full_url = "?".join([base_url, heading_url])

    # Send request
    response = requests.get(full_url)
    # Store response body content as dictionary
    body = json.loads(response.content)

    return response, body

def regulate_api_requests(response: str) -> float:
    """
    Function to adjust time in between API requests to avoid having our requests blocked by PubChem
    Args:
        response: API response
    Returns:
        wait_time: time to wait before sending next request
    """

    # Get throttling statuses as a dataframe
    statuses = parse_throttling_headers(response.headers['X-Throttling-Control'])

    # Set wait time according to status
    if (statuses['status'] == 'green').all():
        wait_time = 0.2
    if (statuses['status'] == 'black').any():
        wait_time = 3600.0
    elif (statuses['status'] == 'red').any():
        wait_time = 60.0
    elif (statuses['status'] == 'yellow').any():
        wait_time = 1.0

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
