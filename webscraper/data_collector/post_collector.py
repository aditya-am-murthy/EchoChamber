import os
import sys
import csv
import json
import argparse
import re
import logging
from datetime import datetime
import time
from tqdm import tqdm

#idea is to take the N most recent posts from donald trump and build a CSV containing all the people that have liked the post as well as commented

# Suppress all logging output - only show progress bar
import sys
from loguru import logger

# Remove default handler and add a null handler
logger.remove()
logger.add(sys.stderr, level="ERROR")  # Only show errors

# Also suppress standard logging
logging.getLogger('tools.api').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)
logging.getLogger('httpcore').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.api import Api  # Now this import should work


def collect_posts(num_posts=1000, max_likes_per_post=500, max_comments_per_post=50, output_file='trump_posts_data.csv', username='realDonaldTrump'):
    """
    Collect posts and interactions
    
    Args:
        num_posts: Number of posts to collect
        max_likes_per_post: Maximum number of likes to collect per post
        max_comments_per_post: Maximum number of comments to collect per post
        output_file: Output CSV file path
        username: Username to collect posts from
    """
    api = Api()
    
    # Backup existing file if it exists
    if os.path.exists(output_file):
        backup_file = f"{output_file}.backup"
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(output_file, backup_file)
        print(f"Backed up existing file to {backup_file}")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['post_id', 'likes', 'comments'])

        count = 0
        errors = 0
        start_time = time.time()
        
        print(f"Starting collection of {num_posts} posts from @{username}...")
        print(f"Max likes per post: {max_likes_per_post}, Max comments per post: {max_comments_per_post}")
        
        # Create progress bar
        pbar = tqdm(total=num_posts, desc="Collecting posts", unit="post")
        
        try:
            for post in api.pull_statuses(username, replies=False):
                if count >= num_posts:
                    break

                try:
                    post_id = post['id']
                    
                    # Fetch liking usernames
                    likes = []
                    try:
                        for user in api.user_likes(post_id, include_all=False, top_num=max_likes_per_post):
                            if 'username' in user:
                                likes.append(user['username'])
                            if len(likes) >= max_likes_per_post:
                                break
                    except Exception as e:
                        pbar.write(f"  Warning: Error fetching likes for post {post_id}: {e}")
                    
                    # Fetch comments with usernames and content (limit to 500)
                    comments = []
                    try:
                        for comment in api.pull_comments(post_id, include_all=False, only_first=False, top_num=max_comments_per_post):
                            if 'account' in comment and 'username' in comment['account']:
                                username = comment['account']['username']
                                # Truncate comment content to reduce file size (max 200 chars)
                                content = comment.get('content', '')
                                # Remove HTML tags and truncate
                                if isinstance(content, str):
                                    content = re.sub(r'<[^>]+>', '', content)  # Remove HTML
                                    content = content[:200] if len(content) > 200 else content  # Truncate to 200 chars
                                else:
                                    content = str(content)[:200]  # Convert to string and truncate
                                comments.append((username, content))
                            if len(comments) >= max_comments_per_post:
                                break
                    except Exception as e:
                        pbar.write(f"  Warning: Error fetching comments for post {post_id}: {e}")

                    # Write the row with lists serialized as JSON
                    writer.writerow([post_id, json.dumps(likes), json.dumps(comments)])
                    f.flush()  # Ensure data is written immediately

                    count += 1
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Likes': len(likes),
                        'Comments': len(comments),
                        'Rate': f'{rate:.2f}/s',
                        'Errors': errors
                    })
                    pbar.update(1)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    errors += 1
                    pbar.write(f"Error processing post: {e}")
                    if errors > 10:
                        pbar.write("Too many errors, stopping collection.")
                        break
                    pbar.update(1)  # Update progress even on error
                    continue
        
        except KeyboardInterrupt:
            pbar.write(f"\nCollection interrupted by user. Collected {count} posts.")
        except Exception as e:
            pbar.write(f"\nCollection stopped due to error: {e}")
            pbar.write(f"Collected {count} posts before error.")
        finally:
            pbar.close()
        
        elapsed = time.time() - start_time
        print(f"\nCollection complete!")
        print(f"  Total posts collected: {count}")
        print(f"  Errors encountered: {errors}")
        print(f"  Time elapsed: {elapsed/60:.2f} minutes")
        print(f"  Average rate: {count/elapsed:.2f} posts/second" if elapsed > 0 else "")
        print(f"  Output file: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Collect posts and interactions from Truth Social')
    parser.add_argument('--num_posts', type=int, default=1000,
                       help='Number of posts to collect (default: 1000)')
    parser.add_argument('--max_likes', type=int, default=500,
                       help='Maximum likes per post to collect (default: 5000)')
    parser.add_argument('--max_comments', type=int, default=50,
                       help='Maximum comments per post to collect (default: 500)')
    parser.add_argument('--output', type=str, default='trump_posts_data.csv',
                       help='Output CSV file path')
    parser.add_argument('--username', type=str, default='realDonaldTrump',
                       help='Username to collect posts from')
    
    args = parser.parse_args()
    
    collect_posts(
        num_posts=args.num_posts,
        max_likes_per_post=args.max_likes,
        max_comments_per_post=args.max_comments,
        output_file=args.output,
        username=args.username
    )


if __name__ == "__main__":
    main()