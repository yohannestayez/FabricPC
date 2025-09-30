import pytest

from models.graph_net import PCGraphNet

pytestmark = [pytest.mark.unit]

def test_duplicate_node_names_raise():
    cfg = {
        "device": "cpu",
        "node_list": [
            {"name": "x", "dim": 2, "type": "linear", "activation": {"type": "sigmoid"}},
            {"name": "x", "dim": 2, "type": "linear", "activation": {"type": "sigmoid"}},
        ],
        "edge_list": [],
        "task_map": {},
    }
    with pytest.raises(ValueError):
        PCGraphNet(config=cfg)

def test_self_edge_disallowed():
    cfg = {
        "device": "cpu",
        "node_list": [
            {"name": "n1", "dim": 2, "type": "linear", "activation": {"type": "sigmoid"}},
        ],
        "edge_list": [
            {"source_name": "n1", "target_name": "n1", "slot": "in"},
        ],
        "task_map": {},
    }
    with pytest.raises(ValueError):
        PCGraphNet(config=cfg)
