import random
from collections import defaultdict
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import LabelSentenceReader, InputExample
from torch.utils.data import DataLoader

def triplets_from_labeled_dataset(input_examples):
    triplets = []
    label2sentence = defaultdict(list)
    for inp_example in input_examples:
        label2sentence[inp_example.label].append(inp_example)

    for inp_example in input_examples:
        anchor = inp_example

        if len(label2sentence[inp_example.label]) < 2:
            continue

        positive = None
        while positive is None or positive.guid == anchor.guid:
            positive = random.choice(label2sentence[inp_example.label])
        
        negative = None
        while negative is None or negative.guid == anchor.guid:
            negative = random.choice(input_examples)

        triplets.append(InputExample(texts=[anchor.texts[0], positive.texts[0], negative.texts[0]]))
    
    return triplets


sbert_model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

sentence_reader = LabelSentenceReader(folder='Data')
data_list = sentence_reader.get_examples(filename='clus_1.tsv')
triplets = triplets_from_labeled_dataset(input_examples=data_list)
finetune_data = SentencesDataset(examples=triplets, model=sbert_model)
finetune_dataloader = DataLoader(finetune_data, shuffle=True)

loss = TripletLoss(model=sbert_model)

out_path = 'bert-base-nli-stsb-mean-tokens-recipes_1'
sbert_model.fit(train_objectives=[(finetune_dataloader, loss)], epochs=4, output_path=out_path)