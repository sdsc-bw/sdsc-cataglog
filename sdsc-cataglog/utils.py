import html2text
import numpy as np
import re
import requests
import openai
import pandas as pd
import os
from tqdm import tqdm

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

def get_embedding(text, model="text-embedding-ada-002"):
    if text is None:
        return None
    text = text.replace("\n", " ")
    return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

def remove_links_from_sentence(sentence):
    pattern = r'\[([^]]+)\]\([^)]+\)'
    cleaned_sentence = re.sub(pattern, '', sentence)

    return cleaned_sentence