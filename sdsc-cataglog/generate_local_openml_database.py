# Exception: The server is overloaded or not ready yet.
import html2text
import numpy as np
import re
import requests
import openai
import openml
import pandas as pd
import os
import time
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from gpt_request import get_response_from_chatgpt_with_context, get_text_embedding_with_openai
from utils import remove_links_from_sentence, delete_sentences_with_high_non_alpha_ratio, clip_text_according_to_token_number
from git_request import search_top_starred_repositories
from tqdm import tqdm


def process_text_from_openml_description_with_openai(dataset_name, dataset_id):
    dataset = openml.datasets.get_dataset(int(dataset_id))
    if dataset:
        md_content = dataset.description
        
        # convert to md
        # md_converter = html2text.HTML2Text()
        # md_converter.ignore_links = True  # 忽略链接
        # md_content = md_converter.handle(readme_content)
        
        # filter according to: 这个很玄学，不知道为什么，但是能过滤掉一些不相关的内容
        tmp = [remove_links_from_sentence(s).strip().replace('\n', '.') for s in md_content.split('\n\n')]
        tmp = [s for s in tmp if '  ' not in s and len(s) > 0]
        tmp = delete_sentences_with_high_non_alpha_ratio('\n'.join(tmp), 0.5).split('\n')
        
        # filter with openai
        out = text_filter_according_to_theme_with_openai(dataset_name, tmp)
        return out
    else:
        print("Failed to retrieve README content.")
    return None

def process_text_from_openml_description_request_with_openai(dataset_name, dataset_id):
    response = requests.get(f'https://www.openml.org/api/v1/data/{dataset_id}')
    if response.status_code == 200:
        root = ET.fromstring(response.text)
        md_content = root.find("{http://openml.org/openml}description").text
        if md_content:
            # convert to md
            # md_converter = html2text.HTML2Text()
            # md_converter.ignore_links = True  # 忽略链接
            # md_content = md_converter.handle(readme_content)

            # filter according to: 这个很玄学，不知道为什么，但是能过滤掉一些不相关的内容
            tmp = [remove_links_from_sentence(s).strip().replace('\n', '.') for s in md_content.split('\n\n')]
            tmp = [s for s in tmp if '  ' not in s and len(s) > 0]
            tmp = delete_sentences_with_high_non_alpha_ratio('\n'.join(tmp), 0.5).split('\n')

            # filter with openai
            out = text_filter_according_to_theme_with_openai(dataset_name, tmp)
            return out
    else:
        print("Failed to retrieve README content.")
    return None

def process_text_from_openml_description_with_bs4(dataset_id):
    dataset = openml.datasets.get_dataset(2)
    text = dataset.description
    
    if text:
        soup = BeautifulSoup(text, 'html.parser')

        # 移除JavaScript和CSS
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text()

        # 移除多余的空白
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
              
        return text
    
    else:
        return None
    
def text_filter_according_to_theme_with_openai(theme, description):
    #prompt = f'Please extract the informatin from the following sentences that related to the {theme}, except about the installation: {description}'
    prompt = f'Summarize the following sentences about the dataset {theme} in 100 words'
    context = []
    prompt = clip_text_according_to_token_number(prompt, 3200) # maximul number depends on the llm model and the number of the tokenize affected by the tokenize method.  
    response, context = get_response_from_chatgpt_with_context(prompt, context)
    return response

def function_summary_with_openai(description):
    prompt = f'Please extract the information about the function according to the following description: {description}'
    context = []
    response, context = get_response_from_chatgpt_with_context(prompt, context)
    return response


def download_and_save_openml_dataset_infomration_with_openml(out_path = './data/openml_dataset_info.csv'):
    """
    这也是使用网络资源生成的embedding，试着使用本地的，看一下耗时。
    """
    start_time = time.time()
    
    datasets = openml.datasets.list_datasets(output_format='dataframe') # get list of dataset
      
    if os.path.exists(out_path):
        df = pd.read_csv(out_path, index_col = 0)
        idx_init = df.shape[0]
    else:
        df = pd.DataFrame(columns = ['name', 'did', 'NumberOfInstances', 'description', 'embedding'])
        idx_init = 0
    
    counter = 0
    for idx in datasets.index: 
        if datasets.loc[idx, 'did'] in df['did'].values:
            continue
            
        else:
            text = process_text_from_openml_description_with_openai(datasets.loc[idx, 'name'], datasets.loc[idx, 'did'])
            if text is None:
                continue

            df.loc[idx_init + counter, 'name'] = datasets.loc[idx, 'name']
            df.loc[idx_init + counter, 'did'] = datasets.loc[idx, 'did']
            df.loc[idx_init + counter, 'NumberOfInstances'] = datasets.loc[idx, 'NumberOfInstances']
            df.loc[idx_init + counter, 'description'] = text

            df.to_csv(out_path)
            counter += 1
            
            if counter == 10:
                break
    
    end_time = time.time()
    print(f"Run time: {end_time - start_time}")
    
    return df

def download_and_save_openml_dataset_infomration_with_request(out_path = './data/openml_dataset_info.csv'):
    """
    这也是使用网络资源生成的embedding，试着使用本地的，看一下耗时。
    """
    start_time = time.time()
    
    datasets = openml.datasets.list_datasets(output_format='dataframe') # get list of dataset
      
    if os.path.exists(out_path):
        df = pd.read_csv(out_path, index_col = 0)
        idx_init = df.shape[0]
    else:
        df = pd.DataFrame(columns = ['name', 'did', 'NumberOfInstances', 'description', 'embedding'])
        idx_init = 0
    
    counter = 0
    for idx in datasets.index: 
        if datasets.loc[idx, 'did'] in df['did'].values:
            continue
            
        else:
            text = process_text_from_openml_description_request_with_openai(datasets.loc[idx, 'name'], datasets.loc[idx, 'did'])
            if text is None:
                continue

            df.loc[idx_init + counter, 'name'] = datasets.loc[idx, 'name']
            df.loc[idx_init + counter, 'did'] = datasets.loc[idx, 'did']
            df.loc[idx_init + counter, 'NumberOfInstances'] = datasets.loc[idx, 'NumberOfInstances']
            df.loc[idx_init + counter, 'description'] = text

            df.to_csv(out_path)
            counter += 1
            
            if counter == 100:
                break
    
    end_time = time.time()
    print(f"Run time: {end_time - start_time}")
    
    return df
