import torch
from torch_geometric.data import HeteroData

from models.gnn import HeteroGNNModel, HeteroGNNConfig


def _build_dummy_hetero():
    hetero = HeteroData()

    # 3 posts, 2 users, 1 tag
    hetero["post"].x = torch.randn(3, 21)
    hetero["user"].x = torch.randn(2, 6)
    hetero["tag"].x = torch.randn(1, 1)

    # Simple edges for a subset of relations
    hetero[("user", "likes", "post")].edge_index = torch.tensor(
        [[0, 1], [0, 2]], dtype=torch.long
    )
    hetero[("post", "precedes", "post")].edge_index = torch.tensor(
        [[0, 1, 1], [1, 2, 0]], dtype=torch.long
    )

    return hetero


def test_hetero_gnn_forward():
    hetero = _build_dummy_hetero()
    metadata = hetero.metadata()

    config = HeteroGNNConfig(
        hidden_dim=16,
        num_layers=2,
        output_dim=8,
        dropout=0.0,
        use_hierarchical_pooling=True,
    )
    model = HeteroGNNModel(config, metadata)

    # Mark user 0 as regular, user 1 as influencer
    user_types = torch.tensor([0, 1], dtype=torch.long)

    out = model(hetero.x_dict, hetero.edge_index_dict, user_types=user_types)

    assert "graph_repr" in out
    assert out["graph_repr"].shape == (config.output_dim,)
    assert out["post_repr"].shape == (config.hidden_dim,)
    assert out["user_regular_repr"].shape == (config.hidden_dim,)
    assert out["user_influencer_repr"].shape == (config.hidden_dim,)
    assert out["tag_repr"].shape == (config.hidden_dim,)


def test_hierarchical_pooling_dimensionality():
    hetero = _build_dummy_hetero()
    metadata = hetero.metadata()

    hidden_dim = 16
    config = HeteroGNNConfig(
        hidden_dim=hidden_dim,
        num_layers=1,
        output_dim=hidden_dim * 4,
        dropout=0.0,
        use_hierarchical_pooling=True,
    )
    model = HeteroGNNModel(config, metadata)

    user_types = torch.tensor([0, 1], dtype=torch.long)
    out = model(hetero.x_dict, hetero.edge_index_dict, user_types=user_types)

    # Graph representation should ultimately come from 4 groups concatenated then projected.
    assert out["graph_repr"].shape == (config.output_dim,)


