from src.tools.tools import web_search , scrape_url
import re


o=web_search.invoke("latest news on AI research")
# for url in o["URL"] :
#     res=scrape_url(url)
#     print("\n------\n")
#     print(res)


urls = re.findall(r'URL:\s*(https?://\S+)', o)
for url in urls :
    result=scrape_url.invoke(url)
    if result.startswith("SCRAPING_FAILED:"):
        print("\n-------\n")
        print(f"Skipping: {url}")
        print("\n-------\n")
    else:
        print("\n-------\n")
        print(result)
        print("\n-------\n")
    
