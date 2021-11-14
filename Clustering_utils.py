import csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from nltk.cluster import kmeans, euclidean_distance

def LSA(matrix, n_components):
    print("LSA!")
    
    lsa = TruncatedSVD(n_components = n_components)
    lsa_matrix = lsa.fit_transform(matrix)
    return lsa_matrix

def KMeans(matrix, n_clusters):
    print("K-means!")

    kmean = kmeans.KMeansClusterer(n_clusters, euclidean_distance, avoid_empty_clusters=True)
    clusters = kmean.cluster(matrix, True)
    return clusters

def T_SNE(matrix):
    print("T-SNE")

    tsne = TSNE(n_components = 2, random_state = 2)
    tsne_matrix = tsne.fit_transform(matrix)
    return tsne_matrix

def plot(clusters, tsne_matrix, pairs):
    print("PLOTTING!")

    pointscluster = pd.DataFrame(
        [
            (id, coords[0], coords[1], cluster)
            for id, coords, cluster in [(id, tsne_matrix[id], clusters[id]) for id in [t_a[0] for t_a in pairs]]
        ],
        columns=["id", "x", "y", "c"]
    )

    pointscluster.plot.scatter(x='x', y='y', c='c', cmap='tab20c', s=10, figsize=(20, 12))
    plt.show()

def merge_cluster_values(clusters, pairs):
    print("MERGING CLUSTER-PAIRS!")

    merge_dicc = {}
    for i in range(len(clusters)):

        id = pairs[i][0]
        trigger = pairs[i][1].text
        ans = pairs[i][2].text

        if clusters[i] in merge_dicc:
            value = merge_dicc[clusters[i]]
            value.append([id, trigger, ans])
            merge_dicc[clusters[i]] = value
        else:
            merge_dicc[clusters[i]] = [[id, trigger, ans]]

    return merge_dicc

# def save_clusters(clusters, pairs, path_to_save, mode = 'w'):
#     print("SAVING CLUSTERS!")
#     merge_dicc = merge_cluster_values(clusters, pairs)

#     i = 0
#     with open(path_to_save, mode) as file:
#         for index in merge_dicc:
#             file.writelines(["Cluster ", str(i), ":\n"])
#             for word in merge_dicc[index]:
#                 file.writelines([str(word[2]), "\n"])
#             file.write('\n\n----------')
#             i += 1

def save_clusters(clusters, pairs, path_to_save, mode = 'w+'):
    print("SAVING CLUSTERS!")
    merge_dicc = merge_cluster_values(clusters, pairs)

    for index in merge_dicc:
        i = 1
        with open('{}/clus_{}.tsv'.format(path_to_save, index), mode) as file:
            tsc_writer = csv.writer(file, delimiter='\t')
            for word in merge_dicc[index]:
                tsc_writer.writerow([i, str(word[1])])
                tsc_writer.writerow([i, str(word[2])])
                i += 1

def save_clusters(clusters, pairs, path_to_save, mode = 'w+'):
    print("SAVING CLUSTERS!")
    merge_dicc = merge_cluster_values(clusters, pairs)

    clus_n = 0
    with open('{}/clus_top_10.tsv'.format(path_to_save), mode) as file:
        i = 1
        tsc_writer = csv.writer(file, delimiter='\t')
        for index in merge_dicc:
            if clus_n < 10:
                for word in merge_dicc[index]:
                    tsc_writer.writerow([i, str(word[1])])
                    tsc_writer.writerow([i, str(word[2])])
                    i += 1
            clus_n += 1
