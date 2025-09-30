import pytest
import torch

from models.graph_net import PCGraphNet

pytestmark = [pytest.mark.smoke]

def test_import_and_instantiate(small_graph_config):
    model = PCGraphNet(config=small_graph_config)
    assert isinstance(model, PCGraphNet)

def test_simple_infer_runs(small_graph_config, device):
    small_graph_config["device"] = str(device)
    model = PCGraphNet(config=small_graph_config)
    B = 2
    x = torch.randn(B, 4, device=device)
    y = torch.randn(B, 3, device=device)
    model.init_latents(clamp_dict={"a": x, "b": y}, batch_size=B, device=device)
    # Should not raise and should update projections/errors
    model.infer(clamps_dict={"a": x, "b": y})
