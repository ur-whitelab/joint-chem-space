from openbabel import openbabel, pybel

from rdkit import Chem
from rdkit.Chem import AllChem

def write_3D_representation(sml, representation='xyz'):
    if representation not in ['xyz', 'pdb', 'gzmat']:
        raise ValueError(f'Unknown representation: {representation}')
    mol = pybel.readstring('smi', sml)
    mol.OBMol.AddHydrogens()
    mol.make3D()
    mol.write(representation, f'obabel.{representation}', overwrite=True)


def get_3D_representation(sml, representation='xyz'):
    if representation not in ['xyz', 'pdb', 'gzmat']:
        raise ValueError(f'Unknown representation: {representation}')
    m=pybel.readstring('smi', sml)
    m.OBMol.AddHydrogens()
    m.make3D()
    conv = openbabel.OBConversion()
    conv.SetOutFormat(representation)
    return conv.WriteString(m.OBMol)

def replace_variables_in_zmatrix(gzmat):
    lines = gzmat.split('\n')

    variables = {}
    for line in lines:
        if "Variables:" in line:
            # Start of variables section
            index = lines.index(line)
            for variable_line in lines[index+1:]:
                if not variable_line:
                    continue
                var_name, var_value = variable_line.split('=')
                variables[var_name.strip()] = var_value.strip()

    new_lines = []
    for line in lines:
        if "Variables:" in line:
            # Start of variables section, end of file rewrite
            break
        for var_name, var_value in variables.items():
            line = line.replace(var_name, var_value)
        new_lines.append(line)
    return '\n'.join(new_lines)

def get_zmat(sml):
    return replace_variables_in_zmatrix(get_3D_representation(sml, representation='gzmat'))