import pandas as pd
import numpy as np

from langchain.embeddings import HuggingFaceInstructEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import torch

from utils import convert_stringlist_to_list

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device})

def search_top_related_local_datasets_with_cs(keyword, database_path = './data/openml_dataset_info.csv'):
    # local database
    df = pd.read_csv(database_path, index_col = 0)
  
    # search similar item with gpt and cos similarity
    #embd_keyword = np.asarray(get_text_embedding_with_openai(query))
    # get top 5 from local database according to cosine similarity with embd_keyword
    embd_database = np.vstack(df[df['embedding'].isna().map(lambda x: not x)]['embedding'].map(lambda x: np.asarray(convert_stringlist_to_list(x)))) #.tolist()
    embd_keyword = np.asarray(get_embedding(keyword))
    similarity = cosine_similarity(embd_keyword.reshape(1, -1), embd_database)
    top_5_index = np.argsort(similarity[0])[-5:][::-1]
    dataset_names = df.loc[top_5_index, 'name'].tolist()
    dataset_ids = df.loc[top_5_index, 'did'].tolist()
    dataset_ids = [int(did) for did in dataset_ids]
    dataset_description = df.loc[top_5_index, 'description'].tolist()
    dataset_urls = [f"https://www.openml.org/d/{did}" for did in dataset_ids]
    return dataset_names, dataset_ids, dataset_description, dataset_urls

def search_top_related_local_repositories_with_cs(keyword, database_path = './data/repositories.csv'):
    # local database
    df = pd.read_csv(database_path, index_col = 0)

    # search similar item with gpt and cos similarity
    #embd_keyword = np.asarray(get_text_embedding_with_openai(query))
    # get top 5 from local database according to cosine similarity with embd_keyword
    embd_database = np.vstack(df[df['embedding'].isna().map(lambda x: not x)]['embedding'].map(lambda x: np.asarray(convert_stringlist_to_list(x)))) #.tolist()
    embd_keyword = np.asarray(get_embedding(keyword))
    similarity = cosine_similarity(embd_keyword.reshape(1, -1), embd_database)
    top_5_index = np.argsort(similarity[0])[-5:][::-1]
    repo_names = df.loc[top_5_index, 'repo_name'].tolist()
    repo_descriptions = df.loc[top_5_index, 'description'].tolist()
    repo_urls = df.loc[top_5_index, 'link'].tolist()
    return repo_names, repo_descriptions, repo_urls

def get_embedding(text):
    emb = embeddings.embed_query(text)
    return emb