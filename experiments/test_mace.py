from mace.modules.models import MACE
import e3nn
import torch
from mace import modules as MACE_modules
from e3nn.util import jit
from mace import data, tools
import numpy as np

atomic_energies = np.array([1.0, 3.0], dtype=float)
table = tools.AtomicNumberTable([1, 8])
config = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.0, -2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
)

config2 = data.Configuration(
    atomic_numbers=np.array([8, 1, 1, 1]),
    positions=np.array(
        [
            [0.0, -2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
)
# Created the rotated environment

def test_mace():
    # Create MACE model
    model_config = dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=MACE_modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=MACE_modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=5,
        num_elements=2,
        hidden_irreps=e3nn.o3.Irreps("32x0e + 32x1o"),
        MLP_irreps=e3nn.o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
    )
    model = MACE_modules.MACE(**model_config)
    #model_compiled = jit.compile(model)

    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config2, z_table=table, cutoff=3.0
    )

    data_loader = tools.torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=1,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    output1 = model(batch.to_dict(), training=True)
    #output2 = model_compiled(batch.to_dict(), training=True)
    #assert torch.allclose(output1["energy"][0], output2["energy"][0])
    #assert torch.allclose(output2["energy"][0], output2["energy"][1])
    return output1

output1 = test_mace()

print(output1['node_energy'][-1].size())
