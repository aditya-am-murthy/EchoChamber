"""
Build a global user-user graph based on interaction patterns
Users who interact with similar posts are connected
"""

import pandas as pd
import json
import numpy as np
import torch
from torch_geometric.data import Data
from collections import defaultdict
import argparse


def build_user_interaction_graph(csv_path: str, top_users: list, min_co_interactions: int = 2):
    """
    Build a user-user graph where edges represent co-interaction patterns
    
    Args:
        csv_path: Path to posts CSV
        top_users: List of top user usernames
        min_co_interactions: Minimum number of posts two users must both interact with to create an edge
    
    Returns:
        PyTorch Geometric Data object representing the user graph
    """
    user_to_idx = {user: idx for idx, user in enumerate(top_users)}
    num_users = len(top_users)
    
    # Track which posts each user interacted with
    user_posts = defaultdict(set)  # {username: set of post_ids}
    
    df = pd.read_csv(csv_path, delimiter=';')
    
    print(f"Building user interaction graph from {len(df)} posts...")
    
    for idx, row in df.iterrows():
        post_id = row['post_id']
        
        try:
            likes = json.loads(row['likes']) if isinstance(row['likes'], str) else []
            comments = json.loads(row['comments']) if isinstance(row['comments'], str) else []
            comment_users = [c[0] if isinstance(c, (list, tuple)) and len(c) > 0 else '' 
                            for c in comments]
            comment_users = [u for u in comment_users if u]
            
            interacted_users = set(likes) | set(comment_users)
            
            # Track interactions for top users only
            for username in interacted_users:
                if username in user_to_idx:
                    user_posts[username].add(post_id)
        except:
            continue
    
    # Build edges: connect users who interact with similar posts
    edges = []
    edge_weights = []
    
    user_list = list(user_to_idx.keys())
    for i, user1 in enumerate(user_list):
        posts1 = user_posts[user1]
        for j, user2 in enumerate(user_list[i+1:], i+1):
            posts2 = user_posts[user2]
            
            # Count co-interactions (posts both users interacted with)
            co_interactions = len(posts1 & posts2)
            
            if co_interactions >= min_co_interactions:
                # Create undirected edge
                edges.append([user_to_idx[user1], user_to_idx[user2]])
                edges.append([user_to_idx[user2], user_to_idx[user1]])
                
                # Weight by number of co-interactions
                weight = min(co_interactions / 10.0, 1.0)  # Normalize
                edge_weights.append(weight)
                edge_weights.append(weight)
    
    if not edges:
        print("Warning: No edges found. Using minimum threshold of 1.")
        # Fallback: connect users with at least 1 co-interaction
        for i, user1 in enumerate(user_list):
            posts1 = user_posts[user1]
            for j, user2 in enumerate(user_list[i+1:], i+1):
                posts2 = user_posts[user2]
                co_interactions = len(posts1 & posts2)
                if co_interactions >= 1:
                    edges.append([user_to_idx[user1], user_to_idx[user2]])
                    edges.append([user_to_idx[user2], user_to_idx[user1]])
                    weight = min(co_interactions / 10.0, 1.0)
                    edge_weights.append(weight)
                    edge_weights.append(weight)
    
    # Create graph
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)
    else:
        # Empty graph
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float)
    
    # Node features: one-hot encoding of user index + interaction count
    node_features = torch.zeros((num_users, num_users + 1))
    for username, user_idx in user_to_idx.items():
        # One-hot encoding
        node_features[user_idx, user_idx] = 1.0
        # Interaction count (normalized)
        interaction_count = len(user_posts[username])
        node_features[user_idx, -1] = min(interaction_count / 100.0, 1.0)
    
    graph = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_users
    )
    
    print(f"Graph built: {num_users} nodes, {len(edges)//2} edges")
    print(f"Average degree: {len(edges) / num_users / 2:.2f}")
    
    return graph, user_to_idx


def main():
    parser = argparse.ArgumentParser(description='Build user interaction graph')
    parser.add_argument('--data', type=str, default='trump_posts_data.csv')
    parser.add_argument('--top_users', type=str, default='top_500_users.csv')
    parser.add_argument('--output', type=str, default='user_graph.pt')
    parser.add_argument('--min_co_interactions', type=int, default=2)
    
    args = parser.parse_args()
    
    # Load top users
    top_users_df = pd.read_csv(args.top_users)
    top_users = top_users_df['username'].tolist()
    
    # Build graph
    graph, user_to_idx = build_user_interaction_graph(
        args.data, 
        top_users, 
        args.min_co_interactions
    )
    
    # Save
    torch.save({
        'graph': graph,
        'user_to_idx': user_to_idx,
        'top_users': top_users
    }, args.output)
    
    print(f"\nSaved graph to {args.output}")


if __name__ == "__main__":
    main()

