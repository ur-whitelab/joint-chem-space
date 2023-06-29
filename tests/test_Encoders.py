import sys
sys.path.append(".")
import chemspace as cs

import pytest
import torch


class TestEncoders:
    @pytest.mark.parametrize('testSMILES',[['CCO'],['CCO','CCO']])
    def test_encoder_shape(self, testSMILES):
        """
        Test that the encoder returns the correct shape.
        Specifically for ChemBERTa-77M-MLM, the output dimension is 384.

        """
        E = cs.Encoder(model_name = "DeepChem/ChemBERTa-77M-MLM")
        batch_size = len(testSMILES)
        test = E.tokenize(testSMILES)
        assert E(test).shape == torch.Tensor(batch_size, 512, 384).shape