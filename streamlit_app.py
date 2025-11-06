import os
import streamlit as st
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.tools import PythonREPLTool
from langchain.tools.retriever import create_retriever_tool
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor

# --------------------------------------------------------------------
# 1. Web Search Tool
# --------------------------------------------------------------------
def search_web():
    # 1. Tavily Search Tool 호출하기
    return TavilySearchResults(k=5)

# --------------------------------------------------------------------
# 1-1. Python REPL Tool
# --------------------------------------------------------------------
def create_python_tool():
    """Python 코드 실행 툴 생성"""
    python_repl = PythonREPLTool()
    python_repl.name = "python_repl"
    python_repl.description = (
        "Python 코드를 실행할 수 있는 도구입니다. "
        "계산, 데이터 분석, 그래프 생성 등이 필요할 때 사용하세요. "
        "입력은 유효한 Python 코드여야 하며, print()를 사용해 결과를 출력하세요."
    )
    return python_repl

# --------------------------------------------------------------------
# 2. PDF Tool
# --------------------------------------------------------------------
def load_pdf_files(uploaded_files):
    # 2. PDF 로더 초기화 및 문서 불러오기
    all_documents = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        all_documents.extend(documents)

    # 3. 텍스트를 일정 단위(chunk)로 분할하기
    #    - chunk_size: 한 덩어리의 최대 길이
    #    - chunk_overlap: 덩어리 간 겹치는 부분 길이
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(all_documents)

    # 4. 분할된 문서들을 임베딩하여 벡터 DB(FAISS)에 저장하기
    vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
    
    # 5. 검색기(retriever) 객체 생성
    retriever = vectorstore.as_retriever()

    # 6. retriever를 LangChain Tool 형태로 변환 -> name은 pdf_search로 지정
    retriever_tool = create_retriever_tool(
        retriever,
        name="pdf_search",
        description="이 도구는 업로드된 PDF 문서에 직접 접근할 수 있게 해줍니다. "
                    "질문이 PDF에서 답변될 수 있을 때는 항상 이 도구를 먼저 사용하세요."
    )
    return retriever_tool


# --------------------------------------------------------------------
# 3. Agent + Prompt 구성
# --------------------------------------------------------------------
def build_agent(tools):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """당신은 기술보증기금(KIBO) 직원들을 돕는 유용한 AI 비서입니다.
        
        다음 규칙을 따라주세요:
        1. PDF가 업로드된 경우, 먼저 항상 'pdf_search' 도구를 사용하세요.
        2. PDF에서 관련 결과를 찾지 못한 경우에만 'tavily_search_results_json' 도구로 웹 검색을 하세요.
        3. 계산, 데이터 분석, 그래프 생성이 필요한 경우 'python_repl' 도구를 사용하세요.
        4. 두 개 이상의 도구를 섞어서 사용하지 마세요. 한 번에 하나의 도구만 사용하세요.
        5. 전문적이고 친근한 톤으로 한국어로 답변하며, 적절한 이모지를 포함하세요.
        6. 답변은 명확하고 구조화되게 작성하세요.
        """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    # 8. agent 및 agent_executor 생성하기
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return agent_executor


# --------------------------------------------------------------------
# 4. Agent 실행 함수 (툴 사용 내역 제거)
# --------------------------------------------------------------------
def ask_agent(agent_executor, question: str):
    result = agent_executor.invoke({"input": question})
    answer = result["output"]
    
    # intermediate_steps에서 마지막만 가져오기
    if result.get("intermediate_steps"):
        last_action, _ = result["intermediate_steps"][-1]
        answer += f"\n\n출처:\n- Tool: {last_action.tool}, Query: {last_action.tool_input}"

    return f"답변:\n{answer}"

# --------------------------------------------------------------------
# 5. Streamlit 메인
# --------------------------------------------------------------------
def main():
    # 10. 여러분의 챗봇에 맞는 스타일로 변경하기
    st.set_page_config(page_title="기술보증기금 AI 비서", layout="wide", page_icon="🤖")
    st.image('data/kibo_image.jpg', width=800)
    st.markdown('---')
    st.title("안녕하세요! RAG + Web을 활용한 '기술보증기금 AI 비서' 입니다")  

    with st.sidebar:
        openai_api = st.text_input("OPENAI API 키", type="password")
        tavily_api = st.text_input("TAVILY API 키", type="password")
        pdf_docs = st.file_uploader("PDF 파일 업로드", accept_multiple_files=True)

    if openai_api and tavily_api:
        os.environ['OPENAI_API_KEY'] = openai_api
        os.environ['TAVILY_API_KEY'] = tavily_api

        tools = [search_web(), create_python_tool()]
        if pdf_docs:
            tools.append(load_pdf_files(pdf_docs))

        agent_executor = build_agent(tools)

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        user_input = st.chat_input("질문을 입력하세요")

        if user_input:
            response = ask_agent(agent_executor, user_input)
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": response})

        for msg in st.session_state["messages"]:
            st.chat_message(msg["role"]).write(msg["content"])

    else:
        st.warning("API 키를 입력하세요.")


if __name__ == "__main__":
    main()