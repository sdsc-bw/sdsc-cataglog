import html2text
import numpy as np
import re
import requests
import openai
import pandas as pd
import os

from gpt_request import get_response_from_chatgpt_with_context
from utils import remove_links_from_sentence, delete_sentences_with_high_non_alpha_ratio, get_embedding
from tqdm import tqdm


def process_text_from_github_readme(repo_names, readme_url):
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


def text_filter_according_to_theme_with_openai(theme, discprition):
    prompt = f'Please extract the following sentence to introduce the {theme}, except about the installation: {discprition}'
    context = []
    response, context = get_response_from_chatgpt_with_context(prompt, context)
    return response

def download_and_save_git_stared_reposiories(username = "cc-king-catalog", out_path = './data/repositories.csv'):
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
            readme_txt = process_text_from_github_readme(repo_name, readme_url)
            embd_txt = get_embedding(readme_txt)
            
            df.loc[i, 'repo_name'] = repo_name
            df.loc[i, 'link'] = repo_url
            df.loc[i, 'description'] = readme_txt
            df.loc[i, 'embedding'] = embd_txt

            df.to_csv(out_path)
                
        return df
    else:
        print("Failed to retrieve starred repositories.")
    return None
