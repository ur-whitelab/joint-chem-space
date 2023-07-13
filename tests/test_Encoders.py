import sys
sys.path.append(".")
import chemspace as cs
from transformers import BertForPreTraining, AutoModelForMaskedLM

import pytest
import torch


class TestEncoders:
    @pytest.mark.parametrize('testSMILES',['OCC', ['CCO'],['CCO','COC']])
    @pytest.mark.parametrize('use_tokens',[True, False])
    def test_sml_encoder_shape(self, testSMILES, use_tokens):
        """
        Test that the encoder returns the correct shape.
        Specifically for ChemBERTa-77M-MLM, the output dimension is 384.

        """
        E = cs.Encoder(model_name = "DeepChem/ChemBERTa-77M-MLM", model_type=AutoModelForMaskedLM)
        batch_size = len(testSMILES) if isinstance(testSMILES, list) else 1
        if use_tokens:
            testSMILES = E.tokenize(testSMILES)
        assert E(testSMILES).shape == torch.Tensor(batch_size, 512, 384).shape

    @pytest.mark.parametrize('testTXT',[['To synthesize CCO, we need to have the following'],
                                        ['CCOHCOO is synthesized using the following chemicals', "CCOHCOO can also be synthesized using the following chemicals, but in a batch"]])
    def test_txt_encoder_shape(self, testTXT):
        """
        Test that the encoder returns the correct shape.
        Specifically for scibert_scivocab_cased, the output dimension is 768.
        """
        E = cs.Encoder(model_name="allenai/scibert_scivocab_cased", model_type=BertForPreTraining)
        batch_size = len(testTXT)
        test = E.tokenize(testTXT)
        assert E(test).shape == torch.Tensor(batch_size, 512, 768).shape