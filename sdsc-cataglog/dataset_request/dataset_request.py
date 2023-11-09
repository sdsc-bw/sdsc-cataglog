import pandas as pd
import numpy as np

from langchain.embeddings import HuggingFaceInstructEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import torch

from dataset_request.utils import convert_stringlist_to_list

device = 'cuda' if torch.cuda.is_available() else 'cpu'
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", model_kwargs={"device": device})

def search_top_related_local_datasets_with_cs(keyword, database_path = './data/openml_dataset_info.csv'):
    """
    Search for the top related local datasets based on cosine similarity.

    Args:
        keyword (str): The keyword to search for related datasets.
        database_path (str): Path to the dataset information CSV file.

    Returns:
        tuple: A tuple containing lists of dataset names, dataset IDs, dataset descriptions, and dataset URLs.
    """
    # Read the local dataset information from a CSV file
    df = pd.read_csv(database_path, index_col = 0)
  
    # Extract embeddings of available datasets from the DataFrame
    embd_database = np.vstack(df[df['embedding'].isna().map(lambda x: not x)]['embedding'].map(lambda x: np.asarray(convert_stringlist_to_list(x))))

    # Get the embedding of the input keyword
    embd_keyword = np.asarray(get_embedding(keyword))

    # Calculate the cosine similarity between the keyword and each dataset
    similarity = cosine_similarity(embd_keyword.reshape(1, -1), embd_database)

    # Find the indices of the top 5 most similar datasets
    top_5_index = np.argsort(similarity[0])[-5:][::-1]

    # Extract information about the top 5 datasets
    dataset_names = df.loc[top_5_index, 'name'].tolist()
    dataset_ids = df.loc[top_5_index, 'did'].tolist()
    dataset_ids = [int(did) for did in dataset_ids]
    dataset_description = df.loc[top_5_index, 'description'].tolist()
    dataset_urls = [f"https://www.openml.org/d/{did}" for did in dataset_ids]
    
    return dataset_names, dataset_ids, dataset_description, dataset_urls

def search_top_related_local_repositories_with_cs(keyword, database_path = './data/repositories.csv'):
    """
    Search for the top related local repositories based on cosine similarity.

    Args:
        keyword (str): The keyword to search for related repositories.
        database_path (str): Path to the repository information CSV file.

    Returns:
        tuple: A tuple containing lists of repository names, repository descriptions, and repository URLs.
    """
    
    # local database
    df = pd.read_csv(database_path, index_col = 0)

    # Extract embeddings of available repositories from the DataFrame
    embd_database = np.vstack(df[df['embedding'].isna().map(lambda x: not x)]['embedding'].map(lambda x: np.asarray(convert_stringlist_to_list(x)))) #.tolist()
    
    # Get the embedding of the input keyword
    embd_keyword = np.asarray(get_embedding(keyword))

    # Calculate the cosine similarity between the keyword and each repository
    similarity = cosine_similarity(embd_keyword.reshape(1, -1), embd_database)

    # Find the indices of the top 5 most similar repositories
    top_5_index = np.argsort(similarity[0])[-5:][::-1]

    # Extract information about the top 5 repositories
    repo_names = df.loc[top_5_index, 'repo_name'].tolist()
    repo_descriptions = df.loc[top_5_index, 'description'].tolist()
    repo_urls = df.loc[top_5_index, 'link'].tolist()
    
    return repo_names, repo_descriptions, repo_urls

def get_embedding(text):
    """
    Get the embedding for a given text using HuggingFace Instruct model.

    Args:
        text (str): The input text for which embedding is to be obtained.

    Returns:
        torch.Tensor: The embedding of the input text.
    """
    emb = embeddings.embed_query(text)
    return emb