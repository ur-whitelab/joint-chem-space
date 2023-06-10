import requests
import time
from typing import Tuple, List


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
            return properties[0]["IUPACName"], properties[0]["CanonicalSMILES"]
        except:
          print(f'Error in downloading data for CID: {cid}')
    return None, None


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
            return data['InformationList']["Information"][1]['Description']
        except IndexError:
            return "No description available."
    else:
        print(f"Failed to get description for compound CID: {cid}")
        return "-"


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
    for cid in range(start_cid, end_cid+1):
        c_name, c_smiles = get_compound_name_and_smiles(cid)
        desc = get_compound_description(cid)
        if c_name is not None:
            names.append(c_name)
            smiless.append(c_smiles)
            descriptions.append(desc)
            print(f"Downloaded compound {cid}")
        else:
            print(f"Failed to download compound {cid}")

        time.sleep(0.2)
        
    return names, smiless, descriptions
