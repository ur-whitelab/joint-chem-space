import sys
sys.path.append(".")
import chemspace as cs

import pytest
import torch
from transformers import AutoModel


class TestProjectors:
    @pytest.mark.parametrize('testSMILES',[['CCO'],['CCO','CCO']])
    def test_projector_shape_sml(self, testSMILES):
        """
        Test that the projector returns the correct shape.
        """
        config = cs.ProjConfig(
            input_size=384, #ChemBERTa-77M-MLM
            output_size=256,
        )
        E = cs.Encoder(model_name="DeepChem/ChemBERTa-77M-MLM") # output shape: (1, 512, 384) 
        P = cs.Projector(**vars(config))
        batch = E.tokenize(testSMILES)
        batch_size = len(testSMILES)
        assert P(E(batch)).shape == torch.Tensor(batch_size, 512, 256).shape


    @pytest.mark.parametrize('testTXT',[['To synthesize CCO, we need to have the following'],['CCOHCOO is synthesized using the following chemicals']])
    def test_projector_shape_txt(self, testTXT):
        """
        Test that the projector returns the correct shape.
        """
        config = cs.ProjConfig(
            input_size=768, #scibert_scivocab_cased
            output_size=256,
        )
        E = cs.Encoder(model_name = "allenai/scibert_scivocab_cased", model_type = AutoModel) # output shape: (1, 512, 768) 
        P = cs.Projector(**vars(config))
        batch = E.tokenize(testTXT)
        batch_size = len(testTXT)
        assert P(E(batch)).shape == torch.Tensor(batch_size, 512, 256).shape