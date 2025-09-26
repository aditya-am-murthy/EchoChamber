from datetime import datetime, timezone
from dateutil import parser as date_parse
from api import Api, LoginErrorException

def as_datetime(date_str):
    """Datetime formatter function. Ensures timezone is UTC. Consider moving to Api class."""
    return date_parse.parse(date_str).replace(tzinfo=timezone.utc)


def print_user_likes():
    api = Api()  # Directly create an instance of Api
    likes = list(api.user_likes(post="115267182780354740", top_num=10))
    
    print("User likes:")
    for like in likes:
        print(like)


if __name__ == "__main__":
    print_user_likes()

