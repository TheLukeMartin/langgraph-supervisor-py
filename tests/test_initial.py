# tests/test_initial.py: Contains test functions

from main import app  # Import the compiled AI system

def test_math_agent():
    """Tests if the math agent correctly performs calculations."""
    assert app.invoke({"messages": [{"role": "user", "content": "What is 2 + 3?"}]})

def test_research_agent():
    """Tests if the research agent returns stock-related data."""
    response = app.invoke({"messages": [{"role": "user", "content": "Tell me about Apple stock."}]})
    assert "Stock data for AAPL" in str(response)

def test_supervisor():
    """Tests if the supervisor correctly routes a stock-related question to the research agent."""
    response = app.invoke({"messages": [{"role": "user", "content": "What's the latest stock price for Tesla?"}]})
    assert "Stock data" in str(response)

