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

#SBATCH --ntasks-per-node=1

#SBATCH --mem-per-cpu=10G

import sys
sys.path.append("/pfs/data5/home/kit/tm/hj7422/2023_sdsc_catalog/sdsc-cataglog/sdsc-cataglog")
import pandas as pd
import numpy as np
import time
from langchain.embeddings import HuggingFaceInstructEmbeddings

def get_docu_embedding_with_localgpt(docu, model_name = "hkunlp/instructor-xl"):
    """
    docu is list of text
    """
    # load embeddings model
    device = 'cpu'
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                               model_kwargs={"device": device})
    
    return [embeddings.embed_query(text) for text in docu]

if __name__ == "__main__":
    df = pd.read_csv('./data/copydata.csv', index_col = 0)
    start_time = time.time()
    get_docu_embedding_with_localgpt(df.iloc[:, 2].values, model_name = "hkunlp/instructor-xl")
    end_time = time.time()
    use_time = end_time - start_time

    print(f"生成embd的总用时是：{use_time}")
    print(f"生成embd的平均用时是：{use_time/df.shape[0]}")
