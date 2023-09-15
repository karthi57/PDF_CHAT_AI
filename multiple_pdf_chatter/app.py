from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()
    # st.set_page_config(page_title="Ask Your pdf")
    # st.header("CHAT WITH YOUR PDF")
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.header("Heyy!! CHAT WITH ME  :books:")

    #Upload file
    pdf = st.file_uploader("Give me a pdf",type="pdf")
    
    #extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text +=page.extract_text()

        #split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
      

        # Check if chunks are empty
        if not chunks:
            st.warning("No text  found in the PDF. May be your pdf contains  text in screenshots/images. Kindly search the text word")
            return

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Check if embeddings can be generated
        try:
            knowledge_base = FAISS.from_texts(chunks, embeddings)
        except Exception as e:
            st.error(f"Error generating embeddings: {e}")
            return


        #show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
              response = chain.run(input_documents=docs, question=user_question)
              print(cb)

            st.write(response)


if __name__=='__main__':
    main()