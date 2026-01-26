import os
import json
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("TAVILY_API_KEY")
if not api_key:
    print("Error: TAVILY_API_KEY not found in environment variables.")

try:
    tavily_client = TavilyClient(api_key=api_key)
except Exception as e:
    print(f"Error initializing TavilyClient: {e}")
    exit(1)

url = "https://pocketbase.io/docs/"
depth = 3

print(f"Starting crawl for {url} with max_depth {depth}...")

try:
    response = tavily_client.crawl(
        url=url,
        max_depth=depth,
        extract_depth="advanced")
    
    print("Crawl completed successfully.")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "crawl_results.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2)
    print("Results saved to crawl_results.json")
    
    if "results" in response:
        print(f"Number of pages crawled: {len(response['results'])}")

except Exception as e:
    print(f"An error occurred during crawl: {e}")
