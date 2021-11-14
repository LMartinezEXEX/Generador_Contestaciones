import re
import emoji
import numpy as np
import pandas as pd
from WordEmbedding import get_w2v_model
from nltk.translate.bleu_score import sentence_bleu
from sklearn.feature_extraction import DictVectorizer

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

    pairs = []
    id = 0
    trigger_ids = [tweet.id for tweet in triggers]
    for tweet in tweets:
        if not tweet.is_trigger and tweet.reply_to_id in trigger_ids:
            trigger = get_tweet_by_id(triggers, tweet.reply_to_id)
            pair = [id, trigger, tweet]
            id += 1
            pairs.append(pair)
    return pairs

def get_matrix(pairs, tweets):
    print("VECTORIZING TWEETS!")

    dict = {}
    model = get_w2v_model(tweets)
    for pair in pairs:
        id = pair[0]
        trigger = pair[1]
        answer = pair[2]

        features = {}

        # STRUCTURAL FEATURES!
        features["A_LEN__"] = len(answer.p_text)
        features["A_URL__"] = answer.urls
        features["A_EMOJI__"] = answer.emojis
        features["A_USERNAME__"] = answer.usernames
        features["A_HASHTAG__"] = answer.hashtags

        features["T_LEN__"] = len(trigger.p_text)
        features["T_URL__"] = trigger.urls
        features["T_EMOJI__"] = trigger.emojis
        features["T_USERNAME__"] = trigger.usernames
        features["T_HASHTAG__"] = trigger.hashtags

        features["TA_BLEU__"] = sentence_bleu(trigger.p_text.split(), answer.p_text.split())

        # CONTENT FEATURES!
            # ANSWER
        a_vector = np.zeros(300)
        a_text = answer.p_text.split()
        for word in a_text:
            wv = np.array(model.wv[word])
            a_vector += wv

            # TRIGGER
        t_text = trigger.p_text.split()
        for word in t_text:
            wv = np.array(model.wv[word])
            a_vector += wv

        for i in range(300):
            features['A_DIM_{}__'.format(i)] = a_vector[i]

        dict[id] = features
    
    vectors = []
    for vect in dict:
        vectors.append(dict[vect])
    
    print("GENERATING MATRIX!")
    dv = DictVectorizer(sparse=False)
    matrix = dv.fit_transform(vectors)

    print("MATRIX DIMENSIONS:", matrix.shape)
    return matrix