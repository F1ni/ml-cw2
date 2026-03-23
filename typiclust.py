from sklearn.cluster import KMeans, MiniBatchKMeans
import numpy as np
# k means clustering

label_budget = 30

features_np = all_features_normalized.numpy()
if label_buget <= 50:
  k_means_algorithm = KMeans(n_clusters=label_budget, random_state=42, n_init='auto')
else:
  k_means_algorithm = MiniBatchKeans(n_clusters=label_budget, random_state=42, n_init='auto')

k_means_algorithm.fit(features_np)

final_typical_images = []

from sklearn.neighbors import NearestNeighbors

cluster_assignments = k_means_algorithm.labels_

nearest = NearestNeighbors(n_neighbors=20, algorithm='brute', n_jobs=-1)

for i in range(label_budget):
    cluster_i = np.where(cluster_assignments == i)[0]

    cluster_features = features_np[cluster_i]

    fit = nearest.fit(cluster_features)

    distance, indicies = fit.kneighbors(cluster_features)


    average_distances = distance.mean(axis=1)

    

    typicality_score = 1 / (average_distances + 1e-8)



    best = np.argmax(typicality_score)

    best_id = cluster_i[best]

    final_typical_images.append(best_id)

    print(f"Cluster {i} winner: Image #{best_id}")


print("All done! Here are your 30 typical images:", final_typical_images)
