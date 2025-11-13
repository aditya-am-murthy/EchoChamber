"""
Find the top N most interactive users from the dataset
"""

import pandas as pd
import json
from collections import Counter
import argparse


def find_top_interactive_users(csv_path: str, top_n: int = 500):
    """
    Find the top N most interactive users
    
    Args:
        csv_path: Path to CSV file
        top_n: Number of top users to return
    
    Returns:
        List of (username, interaction_count) tuples
    """
    user_interactions = Counter()
    
    df = pd.read_csv(csv_path, delimiter=';')
    
    print(f"Processing {len(df)} posts...")
    
    for idx, row in df.iterrows():
        try:
            # Parse likes
            likes = json.loads(row['likes']) if isinstance(row['likes'], str) else []
            for username in likes:
                if username:
                    user_interactions[username] += 1
            
            # Parse comments
            comments = json.loads(row['comments']) if isinstance(row['comments'], str) else []
            for comment in comments:
                if isinstance(comment, (list, tuple)) and len(comment) > 0:
                    username = comment[0]
                    if username:
                        user_interactions[username] += 1
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    # Get top N users
    top_users = user_interactions.most_common(top_n)
    
    print(f"\nTop {len(top_users)} most interactive users:")
    print(f"Total unique users in dataset: {len(user_interactions)}")
    print(f"\nTop 10 users:")
    for i, (username, count) in enumerate(top_users[:10], 1):
        print(f"  {i}. {username}: {count} interactions")
    
    return top_users


def save_top_users(top_users, output_path: str):
    """Save top users to a file"""
    with open(output_path, 'w') as f:
        f.write("username,interaction_count\n")
        for username, count in top_users:
            f.write(f"{username},{count}\n")
    print(f"\nSaved top users to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Find top interactive users')
    parser.add_argument('--data', type=str, default='trump_posts_data.csv',
                       help='Path to CSV file')
    parser.add_argument('--top_n', type=int, default=500,
                       help='Number of top users to find')
    parser.add_argument('--output', type=str, default='top_users.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    top_users = find_top_interactive_users(args.data, args.top_n)
    save_top_users(top_users, args.output)
    
    return top_users


if __name__ == "__main__":
    main()

