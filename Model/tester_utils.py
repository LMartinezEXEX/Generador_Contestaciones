import csv
from nltk.util import pr
import numpy as np
import pandas as pd
from scipy import spatial
from sentence_transformers import SentenceTransformer
from Tweet_utils import *
from nltk.translate.bleu_score import sentence_bleu

def get_fine_tunned_model(model_name):
    return SentenceTransformer(model_name)

def get_trigger_embeddings(model, path, max_pairs):
    encoded_trigger = []
    trigger_arr = []
    answer_arr = []
    n_pair = 0
    with open(path, 'r') as file:
        read_tsv = csv.reader(file, delimiter='\t')

        trigger_line = True
        for row in read_tsv:
            if n_pair == max_pairs:
                break

            if n_pair % 100 == 0:
                print(n_pair)

            if trigger_line:
                trigger = row[1]
                trigger_arr.append(trigger)
                encoded_trigger.append(model.encode(trigger))
                trigger_line = False
            else:
                answer_arr.append(row[1])
                trigger_line = True
                n_pair += 1

    return {'Trigger Embedding': encoded_trigger, 'Trigger Text': trigger_arr, 'Corresponding Answer': answer_arr}

def predict(model, trigger_embeddings, trigger):
    encoded_trigger = model.encode([trigger])

    distances = spatial.distance.cdist(np.array(encoded_trigger), trigger_embeddings['Trigger Embedding'], 'cosine')[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    return (trigger, results[0][0])

def get_answer_prediction(trigger_embeddings, result):
    trigger_arr = trigger_embeddings['Trigger Text']
    answer = trigger_embeddings['Corresponding Answer']
    print("\n\nTRIGGER:\n", result[0], end='\n\n')
    print("CLOSEST TRIGGER:\n", trigger_arr[result[1]], end='\n\n')
    print("ANSWER:\n", answer[result[1]])
    print("-------------------------------------------")

    return answer[result[1]]

def get_trigger_answers(path):
    tweets = get_tweets(path)
    process_tweets(tweets)
    triggers = get_triggers(tweets)
    trigger_answers = get_trigger_answer(tweets, triggers)
    return trigger_answers

def evaluate(sentence, model_response, trigger_answers):
    answers = trigger_answers[sentence]
    answers = [answer.split() for answer in answers]

    print("SE ENCONTRARON", len(answers), "RESPUESTAS!")

    return sentence_bleu(answers, model_response.split(), weights=(1,0,0,0))