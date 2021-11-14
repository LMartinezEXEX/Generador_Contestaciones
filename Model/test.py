import csv
from nltk.util import pr
import numpy as np
from scipy import spatial
from sentence_transformers import SentenceTransformer

recipe_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens-recipes')

encode_trigger = []
trigger_arr = []
answer = []
with open('Data/clus_all.tsv', 'r') as file:
    read_tsv = csv.reader(file, delimiter='\t')

    trigger_line = True
    for row in read_tsv:
        if trigger_line:
            trigger = row[1]
            trigger_arr.append(trigger)
            encode_trigger.append(recipe_model.encode(trigger))
            trigger_line = False
        else:
            answer.append(row[1])
            trigger_line = True

q_a_mapping = {'Trigger Embedding': encode_trigger, 'Trigger Text': trigger_arr, 'Corresponding Answer': answer}


trigger = "La Sinopharm es una vacuna muy buena!"

encoded_trigger = recipe_model.encode([trigger])

distances = spatial.distance.cdist(np.array(encoded_trigger), encode_trigger, 'cosine')[0]
results = zip(range(len(distances)), distances)
results = sorted(results, key=lambda x: x[1])

print('\n\n')
print("TRIGGER:\n", trigger, end='\n\n')
print("CLOSEST TRIGGER:\n", trigger_arr[results[0][0]], end='\n\n')
print("ANSWER:\n", answer[results[0][0]])