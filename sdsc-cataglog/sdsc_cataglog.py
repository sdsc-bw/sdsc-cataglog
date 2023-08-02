from git_request import search_top_starred_repositories, search_top_related_local_repositories_with_cs, search_top_related_local_repositories_with_chroma
from gpt_request import get_response_from_chatgpt_with_context
from generate_local_github_database import download_and_save_git_stared_reposiories_according_to_user, download_and_save_git_reposiories_according_to_keyword
from generate_local_openml_database import download_and_save_openml_dataset_infomration_with_openml, download_and_save_openml_dataset_infomration_with_request
from bs4 import BeautifulSoup

import os
import pandas as pd
import requests

if __name__ == "__main__":
    if False:
        url = "https://github.com/timescale/timescaledb/blob/master/README.md"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = [x.strip() for x in soup.get_text().split('.')]
        print(text)

    # 可能因为overload而返回none，这个问题还没有解决
    #if False:
    # if os.path.exists('./data/repositories.csv'):
    #     df = pd.read_csv('./data/repositories.csv', index_col = 0)
    # else:
    #     #df = download_and_save_git_stared_reposiories_according_to_user("cc-king-catalog", out_path = './data/repositories.csv')
    #df = download_and_save_git_reposiories_according_to_keyword("machine learning", out_path = './data/repositories.csv')
    df = download_and_save_openml_dataset_infomration_with_request()

    if False:
        use_case = input("Please input the use case: ")

        # get the theme from the user case
        prompt = f"What is the theme studied in the following use case, please answer only with a keyword less then 20 letter: {use_case}"
        context = []
        response, context = get_response_from_chatgpt_with_context(prompt, context)
        print(f"\nThe theme of the use case is: {response}\n")

        # get the challenge from the user case
        prompt = f"According to the given description, what is the main problem faced by this study, please answer with 3 keywords and without explanation"
        response, context = get_response_from_chatgpt_with_context(prompt, context)
        print(f"The main challenge facing are: \n{response}\n")

        # ask for keywords for python tools
        prompt = f"I want to search for python tools for the above problem by keywords, what keywords should I use, please give me 3 suggestions and speperate them with semicolon, without explanation"
        response, context = get_response_from_chatgpt_with_context(prompt, context)
        keywords = response.split(";")

        # list the advised git repos
        for keyword in keywords:
            prompt = f"Please explain why data {keyword} is needed in the context of the use case above, and please answer in less than 50 words"
            response, context = get_response_from_chatgpt_with_context(prompt, context)
            print(response)

            #git_urls, readme_urls = search_top_starred_repositories(keyword+' python')
            git_urls, readme_urls = search_top_related_local_repositories_with_cs(keyword, database_path = './data/repositories.csv')
            if git_urls is not None:
                print("For this, we recommend the following tools:")
                for git_url, readme_url in zip(git_urls, readme_urls):
                    print("Repository URL:", git_url)
                    print("README URL:", readme_url)
                print('-'*50)



