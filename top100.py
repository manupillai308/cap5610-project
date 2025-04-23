import numpy as np
import os
import pandas as pd
import tqdm
import sys

dataset = sys.argv[1]
fov = sys.argv[2]

query_path = f"DATASET_{dataset.title()}_Features_{fov}/streetview"
reference_path = f"DATASET_{dataset.title()}_Features_{fov}/bingmap"


ref_ids = np.array(sorted(os.listdir(reference_path)))
reference_features = np.empty((len(ref_ids), 1000))

for i in tqdm.tqdm(range(len(ref_ids))):
    reference_features[i] = np.load(f'{reference_path}/{ref_ids[i]}')

query_ids = np.array(sorted(os.listdir(query_path)))
query_features = np.empty((len(query_ids), 1000))


for i in tqdm.tqdm(range(len(query_ids))):
    query_features[i] = np.load(f'{query_path}/{query_ids[i]}')


query_features_norm = np.sqrt(np.sum(query_features**2, axis=1, keepdims=True))
reference_features_norm = np.sqrt(np.sum(reference_features ** 2, axis=1, keepdims=True))
similarity = np.matmul(query_features/query_features_norm, (reference_features/reference_features_norm).transpose())

similarity = np.argsort(similarity, axis=1)

# np.save('top100.npy', similarity[:, -100:])
with open(f"DATASET_{dataset.title()}_Features_{fov}/top100.txt", 'w') as f:
    for i in tqdm.tqdm(range(len(query_ids))):
        if dataset.lower() == 'vigor':
            sep = "@"
        else:
            sep = ","
        f.write(sep.join([query_ids[i]] + list(ref_ids[similarity[i, -100:]])))
        f.write('\n')
