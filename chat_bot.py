import streamlit as st
import tiktoken  # 텍스트 청크를 나눌 때 토큰 갯수 세는 라이브러리
from loguru import logger  # 행동 로그 기록

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback


def chat_bot():
    st.title("RAG Chat-bot🤖")

    # Streamlit 세션 상태 초기화
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = False

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 문서 관련 궁금한 사항에 대해 문의해주세요!"}]

    # 사이드바 입력
    with st.sidebar:
        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        process = st.button("Process")

    # 파일 처리 및 벡터스토어 생성
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        if uploaded_files:
            files_text = get_text(uploaded_files)
            text_chunks = get_text_chunks(files_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
            st.session_state.processComplete = True
            st.success("Processing complete! You can now ask questions.")
        else:
            st.warning("Please upload at least one file to process.")
            st.stop()

    # 채팅 메시지 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if query := st.chat_input("질문을 입력해주세요."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Assistant 응답 처리
        with st.chat_message("assistant"):
            if st.session_state.conversation:
                with st.spinner("Thinking..."):
                    chain = st.session_state.conversation
                    result = chain({"question": query})
                    response = result.get('answer', "죄송합니다. 답변을 생성할 수 없습니다.")
                    st.markdown(response)

                    # Source documents 출력 (선택적)
                    source_documents = result.get('source_documents', [])
                    if source_documents:
                        with st.expander("참고 문서 확인"):
                            for doc in source_documents[:3]:  # 최대 3개 문서만 표시
                                st.markdown(f"- **Source**: {doc.metadata.get('source', 'N/A')}")
                                st.markdown(f"  {doc.page_content[:200]}...")  # 첫 200자만 표시

                    # Assistant 메시지 추가
                    st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Please process the files first!")


def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))


def get_text(docs):
    doc_list = []
    for doc in docs:
        file_name = doc.name
        with open(file_name, "wb") as file:
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")

        # 파일 유형별 로더 사용
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
        else:
            continue

        documents = loader.load_and_split()
        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100, length_function=tiktoken_len)
    return text_splitter.split_documents(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(text_chunks, embeddings)


def get_conversation_chain(vectorstore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name='gpt-3.5-turbo', temperature=0)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )


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
