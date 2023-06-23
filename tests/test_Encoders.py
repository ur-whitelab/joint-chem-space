import sys
sys.path.append(".")
import chemspace as cs

import pytest
import torch


class TestEncoders:
    def test_encoder_shape(self):
        """
        Test that the encoder returns the correct shape.
        Specifically for ChemBERTa-77M-MLM, the output dimension is 384.

        """
        E = cs.Encoder(model_name = "DeepChem/ChemBERTa-77M-MLM")
        test = ["CCO"]
        assert E(test).shape == torch.Tensor(1, 512, 384).shape
        batch = ["CCO", "CCO"]
        assert E(batch).shape == torch.Tensor(2, 512, 384).shape