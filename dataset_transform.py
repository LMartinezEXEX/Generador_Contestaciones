import re
import csv
import emoji
import pandas as pd

class ALT_Tweet:
    def __init__(self, id, text, reply_to_id, reply_to_user_id, user_id) -> None:
       # self.id = [id]
        self.id = id
        self.user_id = user_id
        self.text = text
        self.reply_to_id = reply_to_id
        self.reply_to_user_id = reply_to_user_id
        self.discarded = False
        self.trigger = False
        self.tmp_ids = []
        self.tri = 0

    def sanitize(self):
        self.text = self.text.replace('\n', ' ')

    def check(self):
        #Check for urls in tweet
        self.urls = re.findall(r'(https?://[^\s]+)', self.text)

        #Check for emojis in tweet
        self.emojis = [emo for emo in self.text if emo in emoji.UNICODE_EMOJI_SPANISH]

        #Check for twitter user mentions
        self.usernames = re.findall(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9-_]+)', self.text)
        if self.usernames and not self.trigger:
            self.usernames.pop(0)

        #Check for hashtags
        self.hashtags = re.findall('(?<=^|(?<=[^a-zA-Z0-9-_\.]))#([A-Za-z]+[A-Za-z0-9-_]+)', self.text)
    
    def to_lower(self):
        self.text.lower()


pf = pd.read_csv('Dataset/tweets.csv')
pf['id_str'] = pf['id_str'].astype(float)
pf['user.id'] = pf['user.id'].astype(float)

tweets = []

#GET TWEETS
for _, row in pf.iterrows():
    id = row['id_str']
    user_id = row['user.id']
    text = row['full_text']
    reply_to_id = row['in_reply_to_status_id']
    reply_to_user_id = row['in_reply_to_user_id']
    tweet = ALT_Tweet(id, text, reply_to_id, reply_to_user_id, user_id)
    tweets.append(tweet)

#MARK TRIGGERS
tri_ids = []
trigs = []
for tweet in tweets:
    if pd.isna(tweet.reply_to_id) and pd.isna(tweet.reply_to_user_id):
        tweet.trigger = True
        tri_ids.append(tweet.id)
        trigs.append(tweet)
        tweet.tri = 1

#GET UTILS
for tweet in tweets:
    tweet.sanitize()
    tweet.check()
    tweet.to_lower()

def find(tweets, id):
    for tweet in tweets:
        if id == tweet.id:
            return tweet

#MERGE THREADS
for tweet in tweets:
    if tweet.reply_to_id in tri_ids:
        trigger_tweet = find(tweets, tweet.reply_to_id)

        if tweet.user_id == trigger_tweet.user_id:
            trigger_tweet.text += " " + tweet.text
            trigger_tweet.urls += tweet.urls
            trigger_tweet.emojis += tweet.emojis
            trigger_tweet.tmp_ids.append(tweet.id)
            trigger_tweet.usernames += tweet.usernames
            trigger_tweet.hashtags += tweet.hashtags

            tweet.discarded = True

for tweet in tweets:
    for tri in trigs:
        if tweet.reply_to_id in tri.tmp_ids:
            tweet.reply_to_id = tri.id

print("WRITING!")
header = ["id", "tweet", "reply_to_id", "reply_to_user_id", "user_id", "username", "emoji", "url", "hashtag", "is_trigger"]
with open('Dataset/processed.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(header)

    for tweet in tweets:
        if not tweet.discarded:
            data = [tweet.id, tweet.text, str(tweet.reply_to_id), str(tweet.reply_to_user_id), str(tweet.user_id), len(tweet.usernames), len(tweet.emojis), len(tweet.urls), len(tweet.hashtags), tweet.tri]
            writer.writerow(data)