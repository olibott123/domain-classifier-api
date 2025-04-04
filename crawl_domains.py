import csv
import requests
import time

API_ENDPOINT = "http://localhost:5001/crawl-and-save"
CSV_PATH = "example_domains.csv"

with open(CSV_PATH, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        domain = row['domain']
        company_type = row['company_type']
        
        print(f"Crawling domain: {domain} ({company_type})")

        try:
            response = requests.post(API_ENDPOINT, json={"url": domain}, timeout=300)
            if response.status_code == 200:
                print(f"Success: {domain}")
            else:
                print(f"Failed ({response.status_code}): {domain}")
        except Exception as e:
            print(f"Error crawling {domain}: {e}")

        time.sleep(5)  # pause between requests
