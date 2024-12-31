import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
from phi.workflow import  RunResponse
from phi.utils.pprint import pprint_run_response
from collections.abc import Iterator
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Agents
websearch_agent = Agent(
    name="WebSearchAgent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True,
    debug=True,
)

financial_agent = Agent(
    name="FinancialAgent",
    role="Get financial information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_news=True,
        )
    ],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
    debug=True,
)

multi_ai_agent = Agent(
    team=[websearch_agent, financial_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    markdown=True,
    debug=True,
)

# Streamlit App
st.set_page_config(page_title="AI Multi-Agent Playground", layout="wide")

# App Title
st.title("AI Multi-Agent Playground")
st.write(
    "Interact with Web Search Agent, Financial Agent, and Multi-Agent to retrieve and summarize data."
)

# Sidebar for Agent Selection
st.sidebar.title("Agent Selection")
agent_options = ["Web Search Agent", "Financial Agent", "Multi-Agent"]
selected_agent = st.sidebar.radio("Choose an agent", agent_options)

# Input Section
st.header("Input")
query = st.text_input("Enter your query:")


# Process the query based on the selected agent
if st.button("Submit"):
    if query.strip() == "":
        st.error("Please enter a query.")
    else:
        with st.spinner("Processing..."):
            try:
                if selected_agent == "Web Search Agent":
                    # response = websearch_agent.print_response(query, stream=True)
                     report_stream: Iterator[RunResponse] = websearch_agent.run(query)
                elif selected_agent == "Financial Agent":
                    # response = financial_agent.print_response(query, stream=True)
                     report_stream: Iterator[RunResponse] = financial_agent.run(query)
                elif selected_agent == "Multi-Agent":
                    # response = multi_ai_agent.print_response(query, stream=True)
                    report_stream: Iterator[RunResponse] = multi_ai_agent.run(query)

                # Display the response
                pprint_run_response(report_stream, markdown=True)
                st.header("Response")
                st.markdown(report_stream)
                

            except Exception as e:
                st.error(f"Error: {e}")