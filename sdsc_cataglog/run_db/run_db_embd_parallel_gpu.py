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

#SBATCH --gres=gpu:2

#SBATCH --ntasks-per-node=1

#SBATCH --mem-per-cpu=10G

import os
#import multiprocessing
#import torch.multiprocessing as mp
import pandas as pd
import numpy as np
import time
import torch
import threading

from langchain.embeddings import HuggingFaceInstructEmbeddings

#from localgpt_request import get_text_embedding_with_localgpt, get_docu_embedding_with_localgpt, get_docu_embedding_save_to_chroma_with_localgpt
#from gpt_request import get_text_embedding_with_openai

            
def worker(docu, scripted_model):
    embeddings_copy = torch.jit.script(model)
    return [embeddings_copy.embed_query(text) for text in docu]


if __name__ == '__main__':
    df = pd.read_csv('./data/copydata.csv', index_col = 0)
    start_time = time.time()
    
     # load embeddings model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                               model_kwargs={"device": device})
    scripted_model = torch.jit.script(model)
    
    # Define the list to be processed
    chunksize = int(df.shape[0]/2)
    threads = []
    for item in np.arange(0, df.shape[0], chunksize):
        docu = df.iloc[item:item+chunksize, 2]
        thread = threading.Thread(target=worker, args = (docu, scripted_model))
        thread.start()
        threads.append(thread)
    
    for thread in threads:
        thread.join()
    # Set the number of parallel processes
    # num_processes = 10

#     # Create a pool of worker processes
#     pool = multiprocessing.Pool(num_processes)

#     # Map the list elements to the processing function
#     pool.map(process_item, my_list)

#     # Close the pool of worker processes
#     pool.close()
#     pool.join()
    
    end_time = time.time()
    use_time = end_time - start_time
    
    print(f"生成embd的总用时是：{use_time}")
    print(f"生成embd的平均用时是：{use_time/df.shape[0]}")

    
    #!/usr/bin/env python
# file: parallel_list_processing.py


