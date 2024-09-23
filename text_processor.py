from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_document(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=10)
    split_docs = text_splitter.split_documents(documents)
    return split_docs
