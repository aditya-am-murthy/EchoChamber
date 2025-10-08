import os
import sys
import csv
import json
from datetime import datetime

#idea is to take the 1000 most recent posts from donald trump and build a CSV containing all the people that have liked the post as well as commented

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.api import Api  # Now this import should work

api = Api()

output_file = 'trump_posts_data.csv'
if os.path.exists(output_file):
    os.remove(output_file)
    print(f"File '{output_file}' deleted successfully.")

with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['post_id', 'likes', 'comments'])

    count = 0
    for post in api.pull_statuses('realDonaldTrump', replies=False):
        if count >= 50:
            break

        post_id = post['id']

        # Fetch up to 1000 liking usernames
        likes = []
        for user in api.user_likes(post_id, include_all=False, top_num=1000):
            likes.append(user['username'])
            if len(likes) >= 50:
                break

        # Fetch up to 1000 comments with usernames and content
        comments = []
        for comment in api.pull_comments(post_id, include_all=False, only_first=False, top_num=1000):
            username = comment['account']['username']
            content = comment['content']
            comments.append((username, content))
            if len(comments) >= 50:
                break

        # Write the row with lists serialized as JSON
        writer.writerow([post_id, json.dumps(likes), json.dumps(comments)])

        count += 1
        print(f"Processed post {count}: {post_id}")

#from there we need to use CUDA to somehow rank the most interactive followers or accounts with Trump.