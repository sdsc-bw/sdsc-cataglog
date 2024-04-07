import html2text
import nltk
import numpy as np
import re
import requests
import openai
import pandas as pd
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
from nltk.tokenize.treebank import TreebankWordDetokenizer


def convert_stringlist_to_list(stringlist):
    return stringlist.replace('[', '').replace(']', '').replace('\'', '').split(', ')

def delete_sentences_with_high_non_alpha_ratio(text, th = 0.5):
    sentences = text.split('\n')  # 将文本拆分为句子
    result = []

    for sentence in sentences:
        cleaned_sentence = re.sub(r'[^a-zA-Z\s]', '', sentence)  # 移除非字母字符
        if len(sentence) == 0:
            continue
        alpha_ratio = len(cleaned_sentence) / len(sentence)  # 计算字母占比
        #print(alpha_ratio, sentence)
        if alpha_ratio >= th:
            result.append(sentence)

    cleaned_text = '\n'.join(result)
    return cleaned_text

def remove_links_from_sentence(sentence):
    pattern = r'\[([^]]+)\]\([^)]+\)'
    cleaned_sentence = re.sub(pattern, '', sentence)

    return cleaned_sentence

def extract_text_from_webfile(file):
    with open(file, "r", encoding='utf-8') as f:
        contents = f.read()

        soup = BeautifulSoup(contents, 'html.parser')

        # 移除JavaScript和CSS
        for script in soup(["script", "style"]):
            script.extract()

        text = soup.get_text()

        # 移除多余的空白
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text


def clip_text_according_to_token_number(text, num):
    # Tokenize the sentence
    tokens = nltk.word_tokenize(text)

    # Print the number of tokens
    if len(tokens) > num:
        tokens = tokens[:num]
    
    reconstructedSentence = TreebankWordDetokenizer().detokenize(tokens)
    
    return reconstructedSentence

def sort_lists(list_1, list_2, list_3, list_4, list_5):
    # Create a dictionary to map items to their indices in list_1
    index = [index for index, item in enumerate(list_1)]
    
    # Sort list_2 based on the order in list_1
    list_2 = sort_list_by_indices(list_2, index)
    list_3 = sort_list_by_indices(list_3, index)
    list_4 = sort_list_by_indices(list_4, index)
    list_5 = sort_list_by_indices(list_5, index)
    
    return list_2, list_3, list_4, list_5

def sort_list_by_indices(data_list, index_list):
    sorted_list = [data_list[i] for i in index_list]
    return sorted_list