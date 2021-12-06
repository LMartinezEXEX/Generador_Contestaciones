import re
import emoji
import numpy as np
import pandas as pd

class Tweet:
    def __init__(self, id, text, reply_to_id, reply_to_user_id, user_id, usernames, emojis, urls, hashtags, is_trigger) -> None:
        self.id = id
        self.user_id = user_id
        self.text = text
        self.reply_to_id = reply_to_id
        self.reply_to_user_id = reply_to_user_id
        self.usernames = usernames
        self.emojis = emojis
        self.urls = urls
        self.hashtags = hashtags
        self.is_trigger = is_trigger

    def process(self):
        # Mask Hashtags
        self.p_text = re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)', 'HASHTAG_TOKEN', self.text)
        # Mask URLs
        self.p_text = re.sub(r'(https?://[^\s]+)', 'URL_TOKEN', self.p_text)
        # Mask digits
        self.p_text = re.sub(r'\d', 'DIGIT_TOKEN', self.p_text)
        # Remove emojis
        self.p_text = ''.join(word for word in self.p_text.split() if not word in emoji.UNICODE_EMOJI_SPANISH)
        #Remove @ from usernames
        self.p_text = re.sub(r'@', '', self.p_text)


def get_tweets(tweets_path):
    print("GETTING TWEETS!")
    df = pd.read_csv(tweets_path)

    tweets = []
    for _, row in df.iterrows():
        id = row['id']
        user_id = row['user_id']
        text = row['tweet']
        reply_to_id = row['reply_to_id']
        reply_to_user_id = row['reply_to_user_id']
        usernames = row['username']
        emojis = row['emoji']
        urls = row['url']
        hashtags = row['hashtag']
        is_trigger = row['is_trigger']
        tweet = Tweet(id, text, reply_to_id, reply_to_user_id, user_id, usernames, emojis, urls, hashtags, is_trigger)
        tweets.append(tweet)

    return tweets

def process_tweets(tweets):
    for tweet in tweets:
        tweet.process()

def get_triggers(tweets):
    print("GETTING TRIGGERS!")

    triggers = []
    for tweet in tweets:
        if tweet.is_trigger:
            triggers.append(tweet)

    return triggers

def get_tweet_by_id(tweets, id):
    for tweet in tweets:
        if tweet.id == id:
            return tweet
    print("NO ENCONTRO TWEET!")

def get_trigger_answer(tweets, triggers):
    print("GETTINS PAIR TRIGGER-ANSWER!")

    pairs = {}
    for tweet in triggers:
        if tweet.text not in pairs:
            pairs[tweet.text] = []

    trigger_ids = [tweet.id for tweet in triggers]
    for tweet in tweets:
        if not tweet.is_trigger and tweet.reply_to_id in trigger_ids:
            trigger = get_tweet_by_id(triggers, tweet.reply_to_id)
            pairs[trigger.text].append(tweet.text)

    return pairs