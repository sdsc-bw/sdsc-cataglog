from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import LlamaTokenizer, LlamaForCausalLM, pipeline
import torch

def get_response_from_localgpt(prompt, model_name = "TheBloke/vicuna-7B-1.1-HF"):
    """
    实现ingest功能，输入是所有的readme 描述
    """
    # load the instructorEmbeddings
    if torch.cuda.is_available():
        device='cuda'
    else:
        device='cpu'
    
    template = """Question: {question}

    Answer: Let's think step by step."""
    
    llm = load_model_for_text_generation(model_name)
    prompt = PromptTemplate(template = template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    print(llm_chain.run(prompt))
    return llm_chain.run(prompt).strip()
    

def get_text_embedding_with_localgpt(text, model_name = "hkunlp/instructor-xl"):
    # load embeddings model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                               model_kwargs={"device": device})
    
    return embeddings.embed_query(text)

def get_docu_embedding_with_localgpt(docu, model_name = "hkunlp/instructor-xl"):
    """
    docu is list of text
    """
    # load embeddings model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                               model_kwargs={"device": device})
    
    return [embeddings.embed_query(text) for text in docu]

def get_docu_embedding_save_to_chroma_with_localgpt(docu, model_name = "hkunlp/instructor-xl"):
    """
    docu is list of text
    """
    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                               model_kwargs={"device": device})

    db = Chroma.from_documents(docu, embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()

def load_model_for_text_generation(model_name = "TheBloke/vicuna-7B-1.1-HF"):
    """
    Select a model on huggingface.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_name)

    model = LlamaForCausalLM.from_pretrained(model_name,
                                             #   load_in_8bit=True, # set these options if your GPU supports them!
                                             #   device_map=1#'auto',
                                             #   torch_dtype=torch.float16,
                                             #   low_cpu_mem_usage=True
                                             )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)

    return local_llm