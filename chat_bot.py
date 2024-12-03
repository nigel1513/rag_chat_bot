import streamlit as st
import tiktoken # 텍스트 청크를 나눌때 토큰 갯수 세는 라이브러리
from loguru import logger # 행동을 취했을 때 구동한 것이 로그로 남기기 위한 라이브러리

from langchain.chains import ConversationalRetrievalChain #
from langchain.chat_models import ChatOpenAI  

from langchain.document_loaders import PyPDFLoader # pdf
from langchain.document_loaders import Docx2txtLoader # word
from langchain.document_loaders import UnstructuredPowerPointLoader 

from langchain.text_splitter import RecursiveCharacterTextSplitter # 나누기
from langchain.embeddings import HuggingFaceEmbeddings # 한국어 특화 임베딩

from langchain.memory import ConversationBufferMemory # 메모리 
from langchain.vectorstores import FAISS # 임시 벡터 저장

from langchain.callbacks import get_openai_callback # 메모리 구현하기 위한 추가 라이브러리
from langchain.memory import StreamlitChatMessageHistory # 메모리 구현하기 위한 추가 라이브러리
def chat_bot():

    st.title("RAG Chat-bot🤖")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
        
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 문서 관련 궁금한 사항에 대해 문의해주세요!"}]


    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    result = chain({"question": query})
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                st.markdown(response)
                #with st.expander("참고 문서 확인"):
                #    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                #    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                #    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    chat_bot()



###################병렬 처리 로직######################################

# def handle_user_query(user_query, dataframe_agent, pdf_search_agent):
#     # 병렬 처리를 위한 에이전트 호출
#     dataframe_response = dataframe_agent(user_query)
#     pdf_response = pdf_search_agent(user_query)
# 
#     # 각 에이전트의 응답을 조합
#     combined_response = combine_responses(dataframe_response, pdf_response)
#     return combined_response
# 
# def combine_responses(dataframe_response, pdf_response):
#     # 여기에 두 응답을 조합하는 로직을 구현
#     # 예: 두 응답을 연결하거나, 특정 조건에 따라 하나의 응답을 우선시
#     return dataframe_response + "\n" + pdf_response
