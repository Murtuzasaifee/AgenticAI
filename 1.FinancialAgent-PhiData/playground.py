from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.storage.agent.sqlite import SqlAgentStorage
from dotenv import load_dotenv
import os
import phi
from phi.playground import Playground, serve_playground_app

## load env variables
load_dotenv()

phi.api = os.getenv("PHI_API_KEY")


## Web Search Agent
websearch_agent = Agent(
    name="WebSearchAgent",
    role="Search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    show_tool_calls=True,
    markdown=True
)

## Financial Agent
financial_agent = Agent(
    name="FinancialAgent",
    role="Get financial information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the data"],
    show_tool_calls=True,
    markdown=True
)

app = Playground(agents=[websearch_agent, financial_agent]).get_app()

if __name__=="__main__":
    serve_playground_app("playground:app", reload=True)
    