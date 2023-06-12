import sys
sys.path.append("..")

import chemspace as cs

class TestPUGUtils:
    def test_dummy_test(self):
        """
        Placeholder test to initialize test suite
        Can be deleted when we have actual tests
        """
        assert True

    def test_get_name_and_compound(self):
        """
        Test that we can retrieve the name and SMILES string for a compound
        """
        cid = 702
        name, smiles = cs.get_compound_name_and_smiles(cid)
        assert isinstance(name, str)
        assert isinstance(smiles, str)
        assert name.lower() == "ethanol"
        assert smiles == "CCO"

    def test_get_compound_description(self):
        """
        Test that we can retrieve the description for a compound
        """
        cid = 702
        description = cs.get_compound_description(cid)
        assert isinstance(description, str)

    def test_download_compound(self):
        """
        Test that we can download a compound
        """
        cid = 701
        compound = cs.download_compounds(cid, cid+1)
        names = compound[0]
        smiless = compound[1]
        descriptions = compound[2]
        assert isinstance(compound, tuple)
        assert isinstance(names, list)
        assert isinstance(smiless, list)
        assert isinstance(descriptions, list)
        assert names[0].lower() == 'ethyl 2-methyl-3-oxobutanoate'
        assert names[1].lower() == 'ethanol'
        assert smiless[0] == "CCOC(=O)C(C)C(=O)C"
        assert smiless[1] == "CCO"
