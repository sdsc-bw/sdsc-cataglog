import html2text
import numpy as np
import re
import requests
import openai
import pandas as pd
import os
import time

from bs4 import BeautifulSoup
from gpt_request import get_response_from_chatgpt_with_context, get_text_embedding_with_openai
from utils import remove_links_from_sentence, delete_sentences_with_high_non_alpha_ratio, clip_text_according_to_token_number
from git_request import search_top_starred_repositories
from tqdm import tqdm


def process_text_from_github_readme_with_openai(repo_names, readme_url):
    response = requests.get(readme_url)
    if response.status_code == 200:
        readme_content = response.text
        
        # convert to md
        md_converter = html2text.HTML2Text()
        md_converter.ignore_links = True  # 忽略链接
        md_content = md_converter.handle(readme_content)
        
        # filter according to: 这个很玄学，不知道为什么，但是能过滤掉一些不相关的内容
        tmp = [remove_links_from_sentence(s).strip().replace('\n', '.') for s in md_content.split('\n\n')]
        tmp = [s for s in tmp if '  ' not in s and len(s) > 0]
        tmp = delete_sentences_with_high_non_alpha_ratio('\n'.join(tmp), 0.5).split('\n')
        
        # filter with openai
        out = text_filter_according_to_theme_with_openai(repo_names, tmp)
        return out
    else:
        print("Failed to retrieve README content.")
    return None

def process_text_from_github_readme_with_bs4(readme_url):
    text = download_github_readme(readme_url)
    
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
    prompt = f'Please extract the following sentence to introduce the {theme}, except about the installation: {description}'
    context = []
    prompt = clip_text_according_to_token_number(prompt, 3200) # maximul number depends on the llm model and the number of the tokenize affected by the tokenize method.  
    response, context = get_response_from_chatgpt_with_context(prompt, context)
    return response

def function_summary_with_openai(description):
    prompt = f'Please extract the information about the function according to the following description: {description}'
    context = []
    response, context = get_response_from_chatgpt_with_context(prompt, context)
    return response

def download_github_readme(url):
    response = requests.get(url)
    if response.status_code == 200:
        readme_content = response.text
        # Save or process the README content as needed
        return readme_content
    else:
        print("Failed to download README. Status code:", response.status_code)
        return None
    
def download_and_save_git_stared_reposiories_according_to_user(username = "cc-king-catalog", out_path = './data/repositories.csv'):
    access_token = "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"

    url = f"https://api.github.com/users/{username}/starred"
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    page = 1
    #response = requests.get(url, headers=headers)
    starred_repos = []
    while True:
        params = {"page": page}
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            page_repos = response.json()
            if len(page_repos) == 0:
                break
            starred_repos.extend(page_repos)
            page += 1
        else:
            print("Failed to retrieve starred repositories.")
            break

    if os.path.exists(out_path):
        df = pd.read_csv(out_path, index_col = 0)
    else:
        df = pd.DataFrame(columns = ['repo_name', 'link', 'description', 'embedding'])

    if len(starred_repos) > 0:
        for i, repo in tqdm(enumerate(starred_repos)):
            if df.loc[i, 'description'] is not np.nan:
                continue
            repo_url = repo["html_url"]
            repo_name = repo["name"]
            readme_url = repo_url + "/blob/master/README.md"
            readme_txt = process_text_from_github_readme_with_openai(repo_name, readme_url)
            embd_txt = get_text_embedding_with_openai(readme_txt)
            
            df.loc[i, 'repo_name'] = repo_name
            df.loc[i, 'link'] = repo_url
            df.loc[i, 'description'] = readme_txt
            df.loc[i, 'embedding'] = embd_txt

            df.to_csv(out_path)
                
        return df
    else:
        print("Failed to retrieve starred repositories.")
    return None

def download_and_save_git_reposiories_according_to_keyword(keyword = "cc-king-catalog", out_path = './data/repositories.csv'):
    """
    这也是使用网络资源生成的embedding，试着使用本地的，看一下耗时。
    """
    start_time = time.time()
    
    repo_urls, readme_urls = search_top_starred_repositories(keyword, num = 10)
    
    if os.path.exists(out_path):
        df = pd.read_csv(out_path, index_col = 0)
        idx_init = df.shape[0]
    else:
        df = pd.DataFrame(columns = ['repo_url', 'readme_url', 'description', 'embedding'])
        idx_init = 0
    
    counter = 0
    for repo_url, readme_url in zip(repo_urls, readme_urls):
        if repo_url in df['repo_url'].values:
            continue
            
        else:
            text = process_text_from_github_readme_with_openai(repo_url.split('/')[-1], readme_url)
            if text is None:
                continue

            df.loc[idx_init + counter, 'repo_url'] = repo_url
            df.loc[idx_init + counter, 'readme_url'] = readme_url
            df.loc[idx_init + counter, 'description'] = text

            df.to_csv(out_path)
            counter += 1
    
    end_time = time.time()
    print(f"Run time: {end_time - start_time}")
    
    return df

def download_and_save_git_reposiories_according_to_keyword_without_embedding(keyword = "cc-king-catalog", num = 10, out_path = './data/repositories.csv'):
    """
    这也是使用网络资源生成的embedding，试着使用本地的，看一下耗时。
    if file exist, append the file 
    """
    start_time = time.time()
    
    repo_urls, readme_urls = search_top_starred_repositories(keyword, num = num)
    
    # check exist
    if os.path.exists(out_path):
        df = pd.read_csv(out_path, index_col = 0)
        idx_init = df.shape[0]
    else:
        df = pd.DataFrame(columns = ['repo_url', 'readme_url', 'description', 'embedding'])
        idx_init = 0
    
    counter = 0
    for repo_url, readme_url in zip(repo_urls, readme_urls):
        if repo_url in df['repo_url'].values:
            continue
        else:
            text = process_text_from_github_readme_with_openai(repo_url.split('/')[-1], readme_url)
            if text is None:
                continue
                
            df.loc[idx_init + counter, 'repo_url'] = repo_url
            df.loc[idx_init + counter, 'readme_url'] = readme_url
            df.loc[idx_init + counter, 'description'] = text

            df.to_csv(out_path)
            counter += 1
    
    end_time = time.time()
    print(f"Run time: {end_time - start_time}")
    
    return df