import sys
sys.path.append(".")
import chemspace as cs

import pytest
import torch


class TestProjectors:
    @pytest.mark.parametrize('testSMILES',[['CCO'],['CCO','CCO']])
    def test_projector_shape(self, testSMILES):
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