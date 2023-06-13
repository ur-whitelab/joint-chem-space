from typing import Union
from openbabel import openbabel, pybel

from rdkit import Chem
from rdkit.Chem import AllChem

def write_3D_representation(
        sml: str, 
        representation: str='xyz',
        filename: str='obabel'
    ) -> None:
    '''
    Writes a 3D representation file of a molecule given its SMILES string.

    Args:
        sml (str): The SMILES string of the molecule.
        representation (str): The desired 3D format ('xyz', 'pdb', or 'gzmat'). Default is 'xyz'.
        filename (str): The name of the file to be written. Default is 'obabel'.

    Raises:
        ValueError: If an unknown representation is provided.
    '''
    if representation not in ['xyz', 'pdb', 'gzmat']:
        raise ValueError(f'Unknown representation: {representation}')
    
    mol = pybel.readstring('smi', sml)
    mol.OBMol.AddHydrogens()
    mol.make3D()
    mol.write(representation, f'{filename}.{representation}', overwrite=True)


def get_3D_representation(
        sml: str, representation: str='xyz'
    ) -> str:
    '''
    Returns a 3D representation of a molecule given its SMILES string.

    Args:
        sml (str): The SMILES string of the molecule.
        representation (str): The desired 3D format ('xyz', 'pdb', or 'gzmat'). Default is 'xyz'.

    Returns:
        str: The 3D representation of the molecule in the requested format.

    Raises:
        ValueError: If an unknown representation is provided.
    '''
    if representation not in ['xyz', 'pdb', 'gzmat']:
        raise ValueError(f'Unknown representation: {representation}')

    m = pybel.readstring('smi', sml)
    m.OBMol.AddHydrogens()
    m.make3D()
    conv = openbabel.OBConversion()
    conv.SetOutFormat(representation)
    return conv.WriteString(m.OBMol)


def replace_variables_in_zmatrix(
        gzmat: str
    ) -> str:
    '''
    Replaces the variables in a Gaussian Z-matrix generated with OpenBabel with their corresponding values.

    Args:
        gzmat (str): The original Gaussian Z-matrix string.

    Returns:
        str: The modified Gaussian Z-matrix with variables replaced by their values.
    '''
    lines = gzmat.split('\n')

    variables = {}
    for line in lines:
        if "Variables:" in line:
            index = lines.index(line)
            for variable_line in lines[index+1:]:
                if not variable_line:
                    continue
                var_name, var_value = variable_line.split('=')
                variables[var_name.strip()] = var_value.strip()

    new_lines = []
    for line in lines:
        if "Variables:" in line:
            break
        for var_name, var_value in variables.items():
            line = line.replace(var_name, var_value)
        new_lines.append(line)
    return '\n'.join(new_lines)


def get_zmat(
        sml: str
    ) -> str:
    '''
    Generates a Gaussian Z-matrix (ZMAT) for a given molecule and replaces the variables 
    with their corresponding values.

    Args:
        sml (str): The SMILES string of the molecule.

    Returns:
        str: The modified ZMAT string of the molecule.
    '''
    return replace_variables_in_zmatrix(get_3D_representation(sml, representation='gzmat'))
