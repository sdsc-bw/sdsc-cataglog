import openml
import pandas as pd

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
import numpy as np

from gpt_request import get_response_from_chatgpt_with_context, get_text_embedding_with_openai
from utils import convert_stringlist_to_list, sort_lists
from sklearn.metrics.pairwise import cosine_similarity
import torch

from tqdm import tqdm



openml.config.apikey = '2e6a8c4cd46e7508cfd5dbeab9e06bdc'



def search_top_related_local_datasets_with_chatgpt(keywords, database_path = './data/openml_dataset_info.csv'):
    # local database
    df = pd.read_csv(database_path, index_col = 0)
    
    dataset_names = []
    dataset_ids = []
    dataset_description = []
    dataset_urls = []
    
    keywords = [f"'{k}'" for k in keywords]
    keywords = ', '.join(keywords)
    
    print(keywords)
    
    context = f"Check if one or multiple of the keywords {keywords} are related to the following dataset description. Only answer with one word 'Yes' or 'No'. \n\n"
    
    print("Searching for datasets...")
    
    for _, row in tqdm(df.iterrows()):
        did = row['did']
        name = row['name']
        description = row['description']
        urls = f"https://www.openml.org/d/{did}"      

        # query design based on keyword
        prompt = context + f"Description: {description}\n\n"
    
        response, _ = get_response_from_chatgpt_with_context(prompt, [])
    
        if response == 'Yes':
            dataset_names.append(name)
            dataset_ids.append(did)
            dataset_description.append(description)
            dataset_urls.append(urls)
    
    return dataset_names, dataset_ids, dataset_description, dataset_urls

def search_top_related_local_datasets_with_chatgpt_ranking(keywords, database_path = './data/openml_dataset_info.csv'):
    # local database
    df = pd.read_csv(database_path, index_col = 0)
    
    dataset_names = []
    dataset_ids = []
    dataset_description = []
    dataset_urls = []
    
    # first find datasets that are related to the keywords
    
    context = f"Check if one or multiple of the keywords '{keywords[0]}', '{keywords[1]}', {keywords[2]} are related to the following dataset description. Only answer with one word 'Yes' or 'No'. \n\n"
    
    print("Searching for datasets...")
    
    for _, row in tqdm(df.iterrows()):
        did = row['did']
        name = row['name']
        description = row['description']
        urls = f"https://www.openml.org/d/{did}"      

        # query design based on keyword
        prompt = context + f"Description: {description}\n\n"
    
        response, _ = get_response_from_chatgpt_with_context(prompt, [])
    
        if response == 'Yes':
            dataset_names.append(name)
            dataset_ids.append(did)
            dataset_description.append(description)
            dataset_urls.append(urls)
            
    # second sort the detected datasets
    
    dataset_names, dataset_ids, dataset_description, dataset_urls = rank_top_related_local_datasets_with_chatgpt(keywords, dataset_names, dataset_ids, dataset_description, dataset_urls)
    
    return dataset_names, dataset_ids, dataset_description, dataset_urls

def rank_top_related_local_datasets_with_chatgpt(keywords, dataset_names, dataset_ids, dataset_description, dataset_urls):
            
    # sort the detected datasets
    
    keywords = [f"'{k}'" for k in keywords]
    keywords = ', '.join(keywords)
    
    prompt = f"Rank the following datasets based on the similarity of the given keywords {keywords}. Only print the IDs starting with the dataset with highest similarity divided by a semicolon without explanation. \n\n"
    
    print("Ranking datasets...")
    
    for did, descriptions in tqdm(zip(dataset_ids, dataset_description)):
        prompt += f"ID: {did}; Description: {descriptions} \n\n"
        
    response, _ = get_response_from_chatgpt_with_context(prompt, [])
    
    ranking = response.split(";")
    for i in range(len(ranking)):
        ranking[i] = int(ranking[i])
    
    dataset_ids, dataset_names, dataset_description, dataset_urls = sort_lists(ranking, dataset_ids, dataset_names, dataset_description, dataset_urls)
    
    return dataset_names, dataset_ids, dataset_description, dataset_urls

def search_top_related_local_datasets_with_cs(keywords, database_path = './data/openml_dataset_info.csv'):
    # local database
    df = pd.read_csv(database_path, index_col = 0)

    # query design based on keyword
    keywords = [f"'{k}'" for k in keywords]
    keywords = ', '.join(keywords)
    #query = f"Search related text based on keywords: {keywords}\n\n"
    
    # search similar item with gpt and cos similarity
    #embd_keyword = np.asarray(get_text_embedding_with_openai(query))
    # get top 5 from local database according to cosine similarity with embd_keyword
    embd_database = np.vstack(df[df['embedding'].isna().map(lambda x: not x)]['embedding'].map(lambda x: np.asarray(convert_stringlist_to_list(x)))) #.tolist()
    embd_keywords = np.asarray(get_embedding(keywords))
    similarity = cosine_similarity(embd_keywords.reshape(1, -1), embd_database)
    top_5_index = np.argsort(similarity[0])[-5:][::-1]
    dataset_names = df.loc[top_5_index, 'name'].tolist()
    dataset_ids = df.loc[top_5_index, 'did'].tolist()
    dataset_description = df.loc[top_5_index, 'description'].tolist()
    dataset_urls = [f"https://www.openml.org/d/{did}" for did in dataset_ids]
    return dataset_names, dataset_ids, dataset_description, dataset_urls

def get_embedding(text, model_name="hkunlp/instructor-xl"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceInstructEmbeddings(model_name=model_name, model_kwargs={"device": device})
    emb = embeddings.embed_query(text)
    return emb


