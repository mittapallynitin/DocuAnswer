import os
from uuid import uuid4

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

VECTOR_STORE = None


def _get_embeddings_model():
    model_source = os.getenv('EMBEDDINGS_MODEL_SOURCE')
    model_name = os.getenv('EMBEDDINGS_MODEL_NAME')
    embeddings = None
    if model_source == "local":
        model_name = model_name or "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    elif model_source == "openai":
        embeddings = OpenAIEmbeddings(model=model_name)
    elif model_source == "fake":
        embeddings = FakeEmbeddings(size=4096)
    else:
        raise ValueError("Unknown model")

    return embeddings


def _create_vector_store():
    global VECTOR_STORE
    embeddings = _get_embeddings_model()
    index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    VECTOR_STORE = vector_store
    return VECTOR_STORE


def add_doc(documents):
    if not VECTOR_STORE:
        _create_vector_store()

    uuids = [str(uuid4()) for _ in range(len(documents))]
    VECTOR_STORE.add_documents(documents=documents, ids=uuids)
    return len(documents)


def get_vector_store():
    if VECTOR_STORE:
        return VECTOR_STORE
    return _create_vector_store()


def get_retriver():
    if VECTOR_STORE:
        return VECTOR_STORE.as_retriever()
    raise ValueError("Retriever not available")


def delete_vector_store():
    global VECTOR_STORE
    VECTOR_STORE = None


def get_vectorstore_size():
    if VECTOR_STORE:
        return len(VECTOR_STORE.index_to_docstore_id)
    return 0


def get_relevant_info(query):
    relevant_docs = VECTOR_STORE.similarity_search(query, k=2)
    relevant_info = "\n".join([doc.page_content for doc in relevant_docs])
    return relevant_info
