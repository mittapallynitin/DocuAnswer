import io

from langchain_core.documents import Document
from pypdf import PdfReader

import database
import llm_generator
import text_processor


def read_pdf(
    filename="local",
    file_path=None,
    bytes_data=None,
    process_tables=False,
    process_pictures=False
):
    if bytes_data:
        reader = PdfReader(io.BytesIO(bytes_data))
    elif file_path:
        reader = PdfReader(file_path)
    else:
        raise ValueError("Provide either file_path or bytes_data to read")

    if process_tables or process_pictures:
        raise NotImplementedError("Provide either process_tables or process")

    n_pages = len(reader.pages)  # Extract text from the PDF
    documents = []
    for page in reader.pages:
        document = Document(page_content=page.extract_text(),
                            metadata={"source": filename}
                            )
        documents.append(document)
    documents = text_processor.split_document(documents)
    n_vectors = database.add_doc(documents)
    return {
        "response": "Documents uploaded successfully",
        "n_pages": n_pages,
        "n_vectors": n_vectors
    }


def get_answer(question):
    response = llm_generator.get_answer(question)
    return response


def on_change_uploaded_file():
    print("Deleting vector store")
    database.delete_vector_store()


def get_vector_store_size():
    return database.get_vectorstore_size()


# if __name__ == "__main__":
#     path1 = "./data/sample.pdf"
#     read_pdf(file_path=path1)
#     prompt, answer = llm_generator.get_answer('What is climate change?"')
#     print(prompt)
#     print("================================================")
#     print(answer)
