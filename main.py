import dotenv
import pandas as pd
import streamlit as st

import authorize
import service

dotenv.load_dotenv()
st.title('Welcome to DocuAnswer 📃📃')

with st.form(key="Login"):
    user_name = st.text_input('Please enter your username')
    password = st.text_input('Please enter your password', type='password')
    sent_password = st.form_submit_button('Authorize')

if authorize.is_authorized(user_name, password):
    st.write('Your are authorized to access')
    uploaded_files = st.file_uploader(
        'Please upload the pdf file',
        on_change=service.on_change_uploaded_file(),
        accept_multiple_files=True)

    if len(uploaded_files):
        pages = []
        vectors_store_size = []
        filenames = []

        for uploaded_file in uploaded_files:
            pdf_byte = uploaded_file.getvalue()
            res = service.read_pdf(bytes_data=pdf_byte,
                                   filename=uploaded_file.name)
            pages.append(res['n_pages'])
            vectors_store_size.append(res['n_vectors'])
            filenames.append(uploaded_file.name)

        upload_summary = pd.DataFrame({
            'pages': pages,
            'vectors_store_size': vectors_store_size,
            'filenames': filenames
        })

        st.table(upload_summary)

        question = st.text_input("Please ask me questions related to document")
        if question:
            response = service.get_answer(question)
            st.write("Answer:")
            st.write(response['answer'])
            st.divider()
            with st.expander("See Prompt Used"):
                st.text(response['prompt'])
            with st.expander("See Relevant docs fetched from vector store"):
                st.text(response['context'])
else:
    st.write("You are not authorized")
