import os

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

import database

llm = None


def parse_relevant_docs(relevant_docs):
    relevant_info = []
    for doc in relevant_docs:
        reference = 'reference:' + doc.metadata['source'] + "\n"
        information = 'information:' + doc.page_content
        relevant_info.append(reference + information)
    return "\n".join(relevant_info)


def get_multi_context(query):
    retriver = database.get_retriver()
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=retriver, llm=llm, include_original=True)
    relevant_docs = retriever_from_llm.invoke(query)
    return parse_relevant_docs(relevant_docs[:2])


def get_context(query):
    vector_store = database.get_vector_store()
    relevant_docs = vector_store.similarity_search(query, k=2)
    return parse_relevant_docs(relevant_docs)


def get_prompt():
    message = """
    Answer this question using the provided context only.
    Context has 2 parts:
    1. reference: source of information
    2. Information: relevant information fetched from the source.
    When answering question, include the source information:

    Example:
    Question: 
    What is most used programming language?
    Context:
    reference: stackoverflow
    information: According to the survey done in 2021, the most used programming language is Python.

    answer: From the reference stackoverflow, the most used programming language is Python

    Donot provide question and context in the response, just provide the answer.


    Question:
    {question}

    Context: 
    {context}
    """
    prompt_template = ChatPromptTemplate.from_messages([("human", message)])
    return prompt_template


def pretty_prompt(prompt_value):
    return "\n".join([message.content for message in prompt_value.messages])


def get_answer(question):
    if not llm:
        initialize_llm()
    prompt_template = get_prompt()
    context = get_multi_context(question)
    inputs = {"context": context, "question": question}
    chain = prompt_template | llm
    prompt = prompt_template.invoke(inputs)
    answer = chain.invoke(inputs).content
    return {'answer': answer, 'prompt': pretty_prompt(prompt), 'context': context}


def initialize_llm():
    global llm
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY'))
