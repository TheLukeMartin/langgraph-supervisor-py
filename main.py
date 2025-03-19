import os
import requests
import yfinance as yf
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

# Load environment variables from .env
load_dotenv()

# Retrieve API Keys
FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
SEC_API_KEY = os.getenv("SEC_API_KEY")

# Initialize OpenAI model
model = ChatOpenAI(model="gpt-4o")

# ================================
# 🚀 Financial Data Retrieval Functions
# ================================

def get_stock_price(ticker: str) -> str:
    """Fetches real-time stock price using multiple APIs (with fallback)."""

    # ✅ First Attempt: Financial Datasets API
    try:
        print(f"🔍 Trying FinancialDatasets.ai for {ticker}...")
        url = "https://api.financialdatasets.ai/prices/historical"
        headers = {"Authorization": f"Bearer {FINANCIAL_DATASETS_API_KEY}"}
        params = {"symbol": ticker, "date_range": "1d"}

        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()
            price = data["data"][0]["close"]
            return f"The latest stock price for {ticker.upper()} is ${price:.2f}"
        else:
            print(f"❌ FinancialDatasets.ai failed: {response.status_code}, {response.text}")
    
    except Exception as e:
        print(f"⚠️ Error with FinancialDatasets.ai: {e}")

    # ✅ Second Attempt: SEC API
    try:
        print(f"🔍 Trying SEC-API for {ticker}...")
        url = f"https://api.sec-api.io/stocks/{ticker}/price"
        headers = {"Authorization": f"Bearer {SEC_API_KEY}"}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            return f"SEC-API: Latest stock price for {ticker.upper()} is ${data['price']:.2f}"
        else:
            print(f"❌ SEC-API failed: {response.status_code}, {response.text}")
    
    except Exception as e:
        print(f"⚠️ Error with SEC-API: {e}")

    # ✅ Final Attempt: Yahoo Finance (yfinance)
    try:
        print(f"🔍 Trying Yahoo Finance for {ticker}...")
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"Yahoo Finance: Latest stock price for {ticker.upper()} is ${price:.2f}"
    
    except Exception as e:
        print(f"⚠️ Error with Yahoo Finance: {e}")

    # If all fail, return None
    return f"❌ All API calls failed for {ticker}."

def get_sec_filings(ticker: str) -> str:
    """Fetches recent SEC filings from SEC-API.io."""
    try:
        print(f"🔍 Fetching SEC filings for {ticker}...")
        url = f"https://api.sec-api.io/filings?ticker={ticker}"
        headers = {"Authorization": f"Bearer {SEC_API_KEY}"}

        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            data = response.json()
            filings = data.get("filings", [])
            if filings:
                latest_filing = filings[0]
                return f"Latest SEC filing for {ticker.upper()}:\n{latest_filing['link']}"
            else:
                return "No recent SEC filings found."
        else:
            return "Failed to retrieve SEC filings."
    
    except Exception as e:
        print(f"⚠️ Error fetching SEC filings: {e}")
        return "Error fetching SEC filings."

# ================================
# 🚀 Creating AI Agents
# ================================

# 📊 Stock Analysis Agent
stock_analysis_agent = create_react_agent(
    model=model,
    tools=[get_stock_price, get_sec_filings],
    name="stock_expert",
    prompt=(
        "You are a financial analyst. When asked about a stock, retrieve the stock price using get_stock_price. "
        "If asked about SEC filings, retrieve them using get_sec_filings. "
        "Provide clear and actionable insights."
    )
)

# 🔢 Math Agent (Placeholder for potential financial calculations)
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert. Always use one tool at a time."
)

# 🌍 Research Agent (Placeholder for stock news, sentiment analysis)
def web_search(query: str) -> str:
    """Simulates a web search (Replace with a stock news API later)."""
    return f"Searching for news on: {query}"

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world-class researcher with access to web search. Do not do any math."
)

# ================================
# 🚀 Creating the Supervisor Agent
# ================================

workflow = create_supervisor(
    [stock_analysis_agent, research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a stock expert, a research expert, and a math expert. "
        "For stock-related questions, use stock_expert. "
        "For general research, use research_agent. "
        "For math problems, use math_agent."
    )
)

# Compile the system
app = workflow.compile()

# ================================
# 🚀 Running a Test Query
# ================================

def run_test():
    """Runs an interactive test query."""
    result = app.invoke({
        "messages": [
            {
                "role": "user",
                "content": "What is Tesla's latest stock price?"
            }
        ]
    })
    print(result)

if __name__ == "__main__":
    run_test()
