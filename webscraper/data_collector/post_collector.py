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
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Try to import wandb, but continue if it's not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

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
    # Initialize wandb (use offline mode if not authenticated)
    wandb_initialized = False
    if WANDB_AVAILABLE:
        # Import wandb locally to avoid scoping issues
        try:
            import wandb as wandb_module
            if wandb_module is not None:
                try:
                    # Check if wandb is authenticated by trying to access the API key
                    api_key = os.getenv('WANDB_API_KEY')
                    if not api_key:
                        # Try to get from wandb settings
                        try:
                            import wandb.settings
                            settings = wandb.settings.Settings()
                            api_key = settings.get('api_key')
                        except Exception:
                            pass
                    
                    if api_key:
                        wandb_module.init(
                            project="truthsocial-scraper",
                            entity="aditya-murthy-ucla",
                            name=f"scrape_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            config={
                                "num_posts": num_posts,
                                "max_likes_per_post": max_likes_per_post,
                                "max_comments_per_post": max_comments_per_post,
                                "username": username,
                                "output_file": output_file
                            }
                        )
                        print("Wandb initialized in online mode. View dashboard at wandb.ai")
                        wandb_initialized = True
                    else:
                        # Fallback to offline mode
                        wandb_module.init(
                            project="truthsocial-scraper",
                            entity="aditya-murthy-ucla",
                            name=f"scrape_{username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            mode="offline",
                            config={
                                "num_posts": num_posts,
                                "max_likes_per_post": max_likes_per_post,
                                "max_comments_per_post": max_comments_per_post,
                                "username": username,
                                "output_file": output_file
                            }
                        )
                        print("Wandb initialized in offline mode. Run 'wandb sync' later to upload logs.")
                        wandb_initialized = True
                except Exception as e:
                    print(f"Warning: Could not initialize wandb: {e}")
                    print("Continuing without wandb logging...")
                    wandb_initialized = False
            else:
                print("Wandb module is None. Continuing without wandb logging...")
                wandb_initialized = False
        except ImportError:
            print("Wandb import failed. Continuing without wandb logging...")
            wandb_initialized = False
    else:
        print("Wandb not available. Continuing without wandb logging...")
        wandb_initialized = False
    
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
        total_likes = 0
        total_comments = 0
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
                    users_processed = 0
                    try:
                        for user in api.user_likes(post_id, include_all=False, top_num=max_likes_per_post):
                            users_processed += 1
                            if 'username' in user:
                                likes.append(user['username'])
                            elif 'acct' in user:  # Alternative field name
                                likes.append(user['acct'])
                            if len(likes) >= max_likes_per_post:
                                break
                        # Debug: log if we got fewer likes than expected
                        if users_processed > 0 and len(likes) < users_processed * 0.5:
                            pbar.write(f"  Note: Post {post_id} - processed {users_processed} users, got {len(likes)} usernames")
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
                    total_likes += len(likes)
                    total_comments += len(comments)
                    elapsed = time.time() - start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    
                    # Log metrics to wandb
                    if wandb_initialized:
                        try:
                            import wandb as wandb_module
                            wandb_module.log({
                                "posts_collected": count,
                                "total_likes": total_likes,
                                "total_comments": total_comments,
                                "avg_likes_per_post": total_likes / count if count > 0 else 0,
                                "avg_comments_per_post": total_comments / count if count > 0 else 0,
                                "errors": errors,
                                "rate_posts_per_sec": rate,
                                "elapsed_time_minutes": elapsed / 60,
                                "progress_percent": (count / num_posts) * 100
                            })
                        except Exception:
                            pass
                    
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
        print(f"  Total likes collected: {total_likes}")
        print(f"  Total comments collected: {total_comments}")
        print(f"  Errors encountered: {errors}")
        print(f"  Time elapsed: {elapsed/60:.2f} minutes")
        print(f"  Average rate: {count/elapsed:.2f} posts/second" if elapsed > 0 else "")
        print(f"  Output file: {output_file}")
        
        # Log final summary to wandb
        if wandb_initialized:
            try:
                import wandb as wandb_module
                wandb_module.log({
                    "final_posts_collected": count,
                    "final_total_likes": total_likes,
                    "final_total_comments": total_comments,
                    "final_errors": errors,
                    "final_elapsed_time_minutes": elapsed / 60,
                    "final_rate_posts_per_sec": count / elapsed if elapsed > 0 else 0
                })
                wandb_module.finish()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description='Collect posts and interactions from Truth Social')
    parser.add_argument('--num_posts', type=int, default=1000,
                       help='Number of posts to collect (default: 1000)')
    parser.add_argument('--max_likes', type=int, default=5000,
                       help='Maximum likes per post to collect (default: 5000)')
    parser.add_argument('--max_comments', type=int, default=1000,
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