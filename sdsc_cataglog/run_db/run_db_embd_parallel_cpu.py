#!/usr/bin/env python

#SBATCH --job-name=cc-king-catalog

#SBATCH --error=/pfs/data5/home/kit/tm/hj7422/2023_sdsc_catalog/sdsc-cataglog/sdsc-cataglog/logs/slurm/%x.%j.err

#SBATCH --output=/pfs/data5/home/kit/tm/hj7422/2023_sdsc_catalog/sdsc-cataglog/sdsc-cataglog/logs/slurm/%x.%j.out

#SBATCH --mail-type=END,FAIL

#SBATCH --mail-user=yhuang@teco.edu

#SBATCH --export=ALL

#SBATCH --time=48:00:00

#SBATCH --partition=sdil

#SBATCH --nodes=1

#SBATCH --cpus-per-task=10

#SBATCH --ntasks-per-node=1

#SBATCH --mem-per-cpu=10G

#SBATCH --gres=gpu:4

#import multiprocessing
#import torch.multiprocessing as mp
import numpy as np
import time
import torch
import threading
import os
os.environ["MODIN_ENGINE"] = "dask"
import modin.pandas as pd

from langchain.embeddings import HuggingFaceInstructEmbeddings

#from localgpt_request import get_text_embedding_with_localgpt, get_docu_embedding_with_localgpt, get_docu_embedding_save_to_chroma_with_localgpt
#from gpt_request import get_text_embedding_with_openai

            
def worker(text):
    #return [embeddings.embed_query(text) for text in docu]
    return embeddings.embed_query(text)

# load here to activate fork, only work for cpu and linux
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                           model_kwargs={"device": 'cpu'})


if __name__ == '__main__':
    df = pd.read_csv('./data/copydata.csv', index_col = 0)
    start_time = time.time()
        
#     chunksize = 1
#     threads = []
#     for item in np.arange(0, df.shape[0], chunksize):
#         docu = df.iloc[item:item+chunksize, 2].values
#         thread = threading.Thread(target=worker, args = (docu,))
#         thread.start()
#         threads.append(thread)
    
#     for thread in threads:
#         thread.join()
    df.iloc[:, 2].map(lambda x: worker(x))
    
    end_time = time.time()
    use_time = end_time - start_time
    
    print(f"生成embd的总用时是：{use_time}")
    print(f"生成embd的平均用时是：{use_time/df.shape[0]}")






