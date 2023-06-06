# import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

RECALL_VALUES = [1, 5, 10, 20]

def eval(args, db_dataloader, q_dataloader, model):
    model = model.eval()
    with torch.no_grad():
        print("Extracting database features from training set...")
        database = []
        database_labels = []
        queries = []
        queries_labels = []

        for images, label, _ in tqdm(db_dataloader):
            descriptors = model(images.to(args.device))
            database.append(descriptors.cpu().numpy())
            database_labels.append(label)
        database = np.concatenate(database, axis=0)
        database_labels = np.concatenate(database_labels, axis=0)


        
        for images, label, _ in tqdm(q_dataloader):
            descriptors = model(images.to(args.device))
            descriptors = descriptors.cpu().numpy()
            queries.append(descriptors)
            queries_labels.append(label)
        queries = np.concatenate(queries, axis=0)
        queries_labels = np.concatenate(queries_labels, axis=0)


        # faiss similarity search
        faiss_index = faiss.IndexFlatL2(args.fc_output_dim)
        faiss_index.add(database)
        del database
        
        print('Calculating recall...')
        # prediction = index
        _, predictions = faiss_index.search(queries, max(RECALL_VALUES))
        
        
        recall_1 = 0
        recall_5 = 0
        recall_10 = 0
        recall_10 = 0
        recall_20 = 0
        
        return recall_1, recall_5, recall_10, recall_20