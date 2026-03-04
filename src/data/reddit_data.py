import praw
import pandas as pd


def fetch_reddit_posts(subreddit, limit=100):

    reddit = praw.Reddit(
        client_id="CLIENT_ID",
        client_secret="CLIENT_SECRET",
        user_agent="financial-sentiment"
    )

    posts = []

    for submission in reddit.subreddit(subreddit).hot(limit=limit):

        posts.append({
            "title": submission.title,
            "score": submission.score,
            "created": submission.created_utc
        })

    return pd.DataFrame(posts)
