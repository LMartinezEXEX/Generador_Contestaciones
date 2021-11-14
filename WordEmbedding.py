from gensim.models import Word2Vec

def get_w2v_model(tweets):
    sentences = []
    for tweet in tweets:
        text_arr = tweet.p_text.split()
        sentences.append(text_arr)
    
    model = Word2Vec(sentences, vector_size=300, min_count=1)

    return model