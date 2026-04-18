from duckduckgo_search import DDGS

print("Testing web connection...")
try:
    with DDGS() as ddgs:
        results = [r for r in ddgs.text("What is the weather in Tokyo today?", max_results=3)]
        print("✅ SUCCESS! Web search is working.")
        print(results)
except Exception as e:
    print(f"❌ FAILED! DuckDuckGo is blocking you or down. Error: {e}")