import numpy as np
import requests
import openai
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from utils import get_embedding, convert_stringlist_to_list

openai.api_key = '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'

def search_top_starred_repositories(keyword):
    url = "https://api.github.com/search/repositories"
    headers = {
        "Accept": "application/vnd.github.v3+json"
    }
    params = {
        "q": keyword,
        "sort": "stars",  # 按星数排序
        "order": "desc",  # 降序排列
        "per_page": 5    # 获取前10个结果
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        repositories = data["items"]
        repo_urls = []
        readme_urls = []
        for repo in repositories:
            repo_url = repo["html_url"]
            repo_urls.append(repo_url)
            readme_urls.append(repo_url + "/blob/master/README.md")
            #stars_count = repo["stargazers_count"]

        return repo_urls, readme_urls
    else:
        print("Error:", response.status_code)
    return None, None



def search_top_related_local_repositories(keyword, database_path = './data/repositories.csv'):
    # local database
    df = pd.read_csv(database_path, index_col = 0)

    # query design based on keyword
    query = f"Search related text based on keyword: {keyword}\n\n"
    
    # chroma
    embeddings = HuggingFaceEmbeddings(model_name = 'hkunlp/instructor-xl', model_kwargs = {"device": 'cpu'})
    db = Chroma.from_text(df['readme'].tolist(), embeddings)
    # search
    embd_keyword = np.asarray(get_embedding(keyword))

    # get top 5 from local database according to cosine similarity with embd_keyword
    #embd_database = np.vstack(df[df['embedding'].isna().map(lambda x: not x)]['embedding'].map(lambda x: np.asarray(convert_stringlist_to_list(x)))) #.tolist()
    #similarity = cosine_similarity(embd_keyword.reshape(1, -1), embd_database)
    #top_5_index = np.argsort(similarity[0])[-5:][::-1]
    #repo_urls = df.loc[top_5_index, 'link'].tolist()
    #readme_urls = [repo_url + "/blob/master/README.md" for repo_url in repo_urls]
    selected = db.similarity_search(embd_keyword, k=5)
    print(selected)
    repo_urls = df.loc[selected, 'link'].tolist()
    readme_urls = [repo_url + "/blob/master/README.md" for repo_url in repo_urls]
    return repo_urls, readme_urls

