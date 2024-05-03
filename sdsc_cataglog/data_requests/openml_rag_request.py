import os
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
    PromptTemplate,
)

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3"
)

# check if storage already exists
PERSIST_DIR = "./storage_openml"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("./data/openml_datasets").load_data()
    index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)], skip_on_failure=True) # with preprocessing -> smaller chunks
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

retriever = index.as_retriever()

def search_top_related_local_datasets(keyword, k=5, score_threshold=0.5):
    query = f"Give me datasets that have to do with the topic '{keyword}'."
    # generate  retrieval results (I used k+2 because sometimes it does not return datasets from the document)
    retriever = index.as_retriever(similarity_top_k=k+2)
    retrieval_results = retriever.retrieve(query)

    # extract results
    d_names = []
    d_id = []
    d_descriptions = []
    for r in retrieval_results:
        text = r.get_text()
        score = r.get_score()

        if score < score_threshold or 'dataset_id' not in text or len(d_names) >= k:
            continue

        text_split = text.split('\r\n', 3)

        d_name = text_split[0].removeprefix('name: ')

        # needs to be checked because sometimes it returns the same datasets multiple times
        if d_name in d_names:
            continue
                                               

        d_names.append(d_name)
        d_id.append(text_split[1].removeprefix('dataset_id: '))
        d_descriptions.append(text_split[2].removeprefix('description: '))

    return d_names, d_id, d_descriptions