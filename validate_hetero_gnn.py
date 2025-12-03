"""
Quick validation script to test the heterogeneous GNN setup.

This script performs a dry run to ensure all components are correctly configured
and can process a small batch of data without errors.

Usage:
    python validate_hetero_gnn.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
import pandas as pd
from models.graph.graph_builder import GraphBuilder
from models.integrated_model import IntegratedNLPGNNModel
from models.gnn.hetero_gnn_model import HeteroGNNConfig

def validate_graph_builder():
    """Test graph builder can create heterogeneous graphs"""
    print("=" * 80)
    print("TESTING GRAPH BUILDER")
    print("=" * 80)
    
    try:
        # Load sample data
        df = pd.read_csv('trump_posts_data.csv', delimiter=';')
        print(f"✓ Loaded {len(df)} posts from CSV")
        
        # Create graph builder
        builder = GraphBuilder(include_news_agencies=False)
        print("✓ Created GraphBuilder instance")
        
        # Build heterogeneous graph for first post
        hetero_graph = builder.build_hetero_graph(df.iloc[[0]])
        print("✓ Built heterogeneous graph")
        
        # Check node types
        print(f"\nNode types: {hetero_graph.node_types}")
        for node_type in hetero_graph.node_types:
            num_nodes = hetero_graph[node_type].x.shape[0]
            num_features = hetero_graph[node_type].x.shape[1]
            print(f"  - {node_type}: {num_nodes} nodes, {num_features} features")
        
        # Check edge types
        print(f"\nEdge types: {len(hetero_graph.edge_types)} types")
        for edge_type in hetero_graph.edge_types:
            num_edges = hetero_graph[edge_type].edge_index.shape[1]
            print(f"  - {edge_type}: {num_edges} edges")
        
        print("\n✓ Graph Builder Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Graph Builder Test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def validate_hetero_gnn():
    """Test heterogeneous GNN model initialization and forward pass"""
    print("=" * 80)
    print("TESTING HETEROGENEOUS GNN MODEL")
    print("=" * 80)
    
    try:
        # Load sample data
        df = pd.read_csv('trump_posts_data.csv', delimiter=';')
        builder = GraphBuilder()
        hetero_graph = builder.build_hetero_graph(df.iloc[[0]])
        
        # Get metadata
        metadata = hetero_graph.metadata()
        print(f"✓ Got graph metadata: {len(metadata[0])} node types, {len(metadata[1])} edge types")
        
        # Create HeteroGNN config
        config = HeteroGNNConfig(
            user_input_dim=6,
            post_input_dim=21,
            tag_input_dim=1,
            hidden_dim=256,
            num_layers=3,
            output_dim=128,
            dropout=0.3
        )
        print("✓ Created HeteroGNNConfig")
        
        # Import HeteroGNNModel
        from models.gnn.hetero_gnn_model import HeteroGNNModel
        
        # Create model
        model = HeteroGNNModel(config, metadata)
        print("✓ Created HeteroGNNModel")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model has {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        # Test forward pass
        x_dict = hetero_graph.x_dict
        edge_index_dict = hetero_graph.edge_index_dict
        
        # Build edge_attr_dict
        edge_attr_dict = {}
        for edge_type in hetero_graph.edge_types:
            data = hetero_graph[edge_type]
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr_dict[edge_type] = data.edge_attr
        
        # Derive user_types from user node features
        user_types = None
        if 'user' in hetero_graph.node_types and hetero_graph['user'].x.numel() > 0:
            user_x = hetero_graph['user'].x
            if user_x.size(1) >= 5:
                influencer_flag = user_x[:, 4]  # Index 4 is influencer flag
                user_types = (influencer_flag > 0.5).long()
        
        print("✓ Prepared input tensors")
        
        # Forward pass
        output = model(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict, user_types=user_types)
        print(f"✓ Forward pass successful")
        
        # Check output
        print(f"\nOutput keys: {output.keys()}")
        print(f"  - graph_repr shape: {output['graph_repr'].shape}")
        print(f"  - post_repr shape: {output['post_repr'].shape}")
        print(f"  - user_regular_repr shape: {output['user_regular_repr'].shape}")
        print(f"  - user_influencer_repr shape: {output['user_influencer_repr'].shape}")
        print(f"  - tag_repr shape: {output['tag_repr'].shape}")
        
        print("\n✓ Heterogeneous GNN Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Heterogeneous GNN Test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def validate_integrated_model():
    """Test integrated NLP + GNN model"""
    print("=" * 80)
    print("TESTING INTEGRATED NLP + GNN MODEL")
    print("=" * 80)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"✓ Using device: {device}")
        
        # Load sample data
        df = pd.read_csv('trump_posts_data.csv', delimiter=';')
        builder = GraphBuilder()
        hetero_graph = builder.build_hetero_graph(df.iloc[[0]])
        
        # Create integrated model
        from models.gnn.gnn_model import GNNConfig
        
        gnn_config = GNNConfig(
            input_dim=7,
            hidden_dim=256,
            num_layers=3,
            output_dim=128,
            dropout=0.3
        )
        
        model = IntegratedNLPGNNModel(
            gnn_config=gnn_config,
            use_hetero_gnn=True,
            include_news_context=False,
            device=device
        ).to(device)
        print("✓ Created IntegratedNLPGNNModel")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model has {total_params:,} parameters")
        
        # Test forward pass
        post_text = "Test post content for validation"
        hetero_graph = hetero_graph.to(device)
        
        print("✓ Prepared inputs")
        
        # Forward pass
        virality_score = model(post_text, hetero_graph, news_context=None)
        print(f"✓ Forward pass successful")
        print(f"  - Predicted virality score: {virality_score.item():.4f}")
        
        # Test predict method
        score = model.predict(post_text, hetero_graph)
        print(f"✓ Predict method successful")
        print(f"  - Predicted score: {score:.4f}")
        
        print("\n✓ Integrated Model Test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n✗ Integrated Model Test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "HETEROGENEOUS GNN VALIDATION" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    results = []
    
    # Test 1: Graph Builder
    results.append(("Graph Builder", validate_graph_builder()))
    
    # Test 2: Heterogeneous GNN
    results.append(("Heterogeneous GNN", validate_hetero_gnn()))
    
    # Test 3: Integrated Model
    results.append(("Integrated Model", validate_integrated_model()))
    
    # Summary
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Your heterogeneous GNN is ready to use.")
        print("\nNext steps:")
        print("  1. Run training: python train.py --epochs 10 --batch_size 4")
        print("  2. Monitor training metrics (Loss, MAE, R², Accuracy)")
        print("  3. Check saved model in checkpoints/best_model.pt")
        print("  4. Read HETEROGENEOUS_GNN_GUIDE.md for detailed usage\n")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED. Please check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("  2. Check your data file exists: trump_posts_data.csv")
        print("  3. Verify CSV format uses ';' delimiter")
        print("  4. Read HETEROGENEOUS_GNN_GUIDE.md troubleshooting section\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
