import torch
import pytest

# Configuration file for test suite

@pytest.fixture(scope="session")
def device():
    # Keep tests portable across machines
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def small_graph_config():
    # Minimal directed chain: a -> b
    return {
        "device": "cpu",
        "node_list": [
            {"name": "a", "dim": 4, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "b", "dim": 3, "type": "linear", "activation": {"type": "sigmoid"}},
        ],
        "edge_list": [
            {"source_name": "a", "target_name": "b", "slot": "in"},
        ],
        "task_map": {"x": "a", "y": "b"},
        "T_infer": 2,
        "eta_infer": 0.05,
        "optimizer": {"type": "adam", "lr": 1e-3},
    }
