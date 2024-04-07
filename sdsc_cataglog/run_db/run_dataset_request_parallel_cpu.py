#!/usr/bin/env python

#SBATCH --job-name=cc-king-catalog

#SBATCH --error=/pfs/data5/home/kit/tm/hj7422/2023_sdsc_catalog/sdsc-cataglog/sdsc-cataglog/logs/slurm/%x.%j.err

#SBATCH --output=/pfs/data5/home/kit/tm/hj7422/2023_sdsc_catalog/sdsc-cataglog/sdsc-cataglog/logs/slurm/%x.%j.out

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=yhuang@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

#SBATCH --mem-per-cpu=10G

import sys
sys.path.append("/pfs/data5/home/kit/tm/hj7422/2023_sdsc_catalog/sdsc-cataglog/sdsc-cataglog")

import pandas as pd

from langchain.embeddings import HuggingFaceInstructEmbeddings
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
import torch

import time
from tqdm import tqdm
from multiprocessing import Pool

from utils import convert_stringlist_to_list


device = 'cpu'
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device})

def worker(keywords, database_path='./data/openml_dataset_info.csv'):
    # local database
    df = pd.read_csv(database_path, index_col = 0)

    # query design based on keyword
    keywords = [f"'{k}'" for k in keywords]
    keywords = ', '.join(keywords)
    
    # search similar item with gpt and cos similarity
    embd_database = np.vstack(df[df['embedding'].isna().map(lambda x: not x)]['embedding'].map(lambda x: np.asarray(convert_stringlist_to_list(x)))) #.tolist()
    embd_keywords = np.asarray(get_embedding(keywords))
    similarity = cosine_similarity(embd_keywords.reshape(1, -1), embd_database)
    top_5_index = np.argsort(similarity[0])[-5:][::-1]
    dataset_names = df.loc[top_5_index, 'name'].tolist()
    dataset_ids = df.loc[top_5_index, 'did'].tolist()
    dataset_description = df.loc[top_5_index, 'description'].tolist()
    dataset_urls = [f"https://www.openml.org/d/{did}" for did in dataset_ids]
    return dataset_names, dataset_ids, dataset_description, dataset_urls

# Define the compute_similarity function outside
def compute_similarity(df, get_embedding, keyword):
    keyword = [f"'{k}'" for k in keyword]
    keyword = ', '.join(keyword)

    embd_database = np.vstack(df[df['embedding'].isna().map(lambda x: not x)]['embedding'].map(lambda x: np.asarray(convert_stringlist_to_list(x))))
    embd_keywords = np.asarray(get_embedding(keyword))
    similarity = cosine_similarity(embd_keywords.reshape(1, -1), embd_database)
    top_5_index = np.argsort(similarity[0])[-5:][::-1]
    dataset_names = df.loc[top_5_index, 'name'].tolist()
    dataset_ids = df.loc[top_5_index, 'did'].tolist()
    dataset_description = df.loc[top_5_index, 'description'].tolist()
    dataset_urls = [f"https://www.openml.org/d/{did}" for did in dataset_ids]
    return dataset_names, dataset_ids, dataset_description, dataset_urls

def get_embedding(text):
    emb = embeddings.embed_query(text)
    return emb

if __name__ == '__main__':
    with Pool(5) as p:
        df = pd.read_csv('./data/copydata.csv', index_col = 0)
        text = df.iloc[:, 2]
        keywords = text.strip('][').split(', ') # how get the keywords?
        start_time = time.time()

        p.map(worker, keywords)

        end_time = time.time()
        use_time = end_time - start_time

        print(f"生成embd的总用时是：{use_time}")
        print(f"生成embd的平均用时是：{use_time/df.shape[0]}")


