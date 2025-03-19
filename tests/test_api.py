# %%
import requests
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()
FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")

# ✅ Corrected API URL
url = "https://api.financialdatasets.ai/v1/prices/historical"
headers = {"Authorization": f"Bearer {FINANCIAL_DATASETS_API_KEY}"}
params = {
    "symbol": "TSLA",
    "date_range": "1d"
}

# 🔍 Make the request
response = requests.get(url, headers=headers, params=params)

# 🔎 Debugging output
print("🔍 Status Code:", response.status_code)
print("🔍 Response Text:", response.text)

# %%
