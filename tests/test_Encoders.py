import sys
sys.path.append(".")
import chemspace as cs
from transformers import AutoModel

import pytest
import torch


class TestEncoders:
    @pytest.mark.parametrize('testSMILES',[['CCO'],['CCO','CCO']])
    def test_sml_encoder_shape(self, testSMILES):
        """
        Test that the encoder returns the correct shape.
        Specifically for ChemBERTa-77M-MLM, the output dimension is 384.

        """
        E = cs.Encoder(model_name = "DeepChem/ChemBERTa-77M-MLM")
        batch_size = len(testSMILES)
        test = E.tokenize(testSMILES)
        assert E(test).shape == torch.Tensor(batch_size, 512, 384).shape

    @pytest.mark.parametrize('testTXT',[['To synthesize CCO, we need to have the following'],['CCOHCOO is synthesized using the following chemicals']])
    def test_txt_encoder_shape(self, testTXT):
        """
        Test that the encoder returns the correct shape.
        Specifically for scibert_scivocab_cased, the output dimension is 768.
        """
        E = cs.Encoder(model_name="allenai/scibert_scivocab_cased", model_type=AutoModel)
        batch_size = len(testTXT)
        test = E.tokenize(testTXT)
        assert E(test).shape == torch.Tensor(batch_size, 512, 768).shape