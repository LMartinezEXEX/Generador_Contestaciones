from Clustering_utils import *
from Tweet_utils import *

def clustering():
    tweets = get_tweets('Dataset/processed.csv')

    process_tweets(tweets)

    triggers = get_triggers(tweets)

    pairs = get_trigger_answer(tweets, triggers)

    matrix = get_matrix(pairs, tweets)

    matrix = LSA(matrix, 100)

    clusters = KMeans(matrix, 100)

    tsne_matrix = T_SNE(matrix)

    save_clusters(clusters, pairs, 'Resultados')

    plot(clusters, tsne_matrix, pairs)

if __name__ == "__main__":
    clustering()