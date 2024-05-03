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
PERSIST_DIR = "./storage_github"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("./data/github_repositories").load_data()
    index = VectorStoreIndex.from_documents(documents, transformations=[SentenceSplitter(chunk_size=256)], skip_on_failure=True) # with preprocessing -> smaller chunks
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

retriever = index.as_retriever()

def search_top_related_local_repositories(keyword, k=5, score_threshold=0.5):
    query = f"Give me repositories that have to do with the topic '{keyword}'."
    # generate  retrieval results (I used k+2 because sometimes it does not return repositries from the document)
    retriever = index.as_retriever(similarity_top_k=k+2)
    retrieval_results = retriever.retrieve(query)

    # extract results
    repo_names = []
    repo_urls = []
    repo_descriptions = []
    for r in retrieval_results:
        text = r.get_text()
        score = r.get_score()

        if score < score_threshold or 'repo_name' not in text or len(repo_names) >= k:
            continue

        text_split = text.split('\r\n', 3)

        repo_name = text_split[0].removeprefix('repo_name: ')

        # needs to be checked because sometimes it returns the same repo multiple times
        if repo_name in repo_names:
            continue
                                               

        repo_names.append(repo_name)
        repo_urls.append(text_split[1].removeprefix('link: '))
        repo_descriptions.append(text_split[2].removeprefix('description: '))

    return repo_names, repo_urls, repo_descriptions