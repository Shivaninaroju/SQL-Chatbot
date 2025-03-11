import streamlit as st
from pathlib import Path
from langchain.sql_database import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.callbacks import StreamlitCallbackHandler  
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜")
st.title("🦜 Langchain: Chat with SQL DB")

# Database options
DB_LOCAL = "LOCAL_SQLITE"
DB_MYSQL = "MYSQL"

db_options = ["Use SQLite 3 Database - Employee.db", "Connect to MySQL Database"]
db_choice = st.sidebar.radio("Choose Database", options=db_options)

db_uri = DB_LOCAL if db_options.index(db_choice) == 0 else DB_MYSQL

if db_uri == DB_MYSQL:
    mysql_host = st.sidebar.text_input("MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    mysql_host = mysql_user = mysql_password = mysql_db = None

api_key = st.sidebar.text_input("Groq API Key", type="password", value=GROQ_API_KEY or "")

if not api_key:
    st.warning("Please provide a valid Groq API key.")
    st.stop()

# Initialize LLM model
llm = ChatGroq(groq_api_key=api_key, model_name="llama3-8b-8192", streaming=False)  # Disabled streaming for faster response

@st.cache_resource(ttl=7200)
def configure_db(db_uri, host=None, user=None, password=None, db_name=None):
    if db_uri == DB_LOCAL:
        db_path = (Path(__file__).parent / "Employee.db").absolute()
        return SQLDatabase(create_engine("sqlite://", creator=lambda: sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)))
    elif db_uri == DB_MYSQL:
        if not all([host, user, password, db_name]):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{user}:{password}@{host}/{db_name}"))
    else:
        st.error("Invalid Database Selection")
        st.stop()

# Set up database connection
db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)

# Toolkit setup
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=False)  # Disabled verbosity for faster execution

# Chat History Management
if "messages" not in st.session_state or st.sidebar.button("Clear Chat History"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I assist you today?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
user_query = st.chat_input("Ask the database...")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message("assistant"):
        try:
            callback = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[callback])
            if not response or response.strip().lower() in ["o'connell", "no data found", "none", "null"]:
                response = "No data found in the database."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
        except Exception:
            response = "No data found in the database."
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
