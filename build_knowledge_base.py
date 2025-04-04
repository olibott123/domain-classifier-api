import csv
import requests
import time
import os

API_ENDPOINT = "http://localhost:5001/crawl-and-save"
INPUT_CSV = "example_domains.csv"
OUTPUT_CSV = "knowledge_base.csv"

with open(INPUT_CSV, 'r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    domains = list(reader)

if not os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as kb_file:
        writer = csv.writer(kb_file)
        writer.writerow(['domain', 'company_type', 'content'])

for entry in domains:
    domain = entry['domain']
    company_type = entry['company_type']

    print(f"Crawling domain: {domain} ({company_type})")
    try:
        response = requests.post(API_ENDPOINT, json={"url": domain}, timeout=600)
        result = response.json()

        if response.status_code == 200 and result.get('success'):
            content_snippet = result.get('content_snippet', '')
            print(f"Success: {domain}")

            # Write to local KB CSV
            with open(OUTPUT_CSV, 'a', encoding='utf-8', newline='') as kb_file:
                writer = csv.writer(kb_file)
                writer.writerow([domain, company_type, content_snippet])

        else:
            print(f"Failed ({response.status_code}): {domain} - {result.get('error')}")

    except Exception as e:
        print(f"Exception crawling {domain}: {e}")

    time.sleep(10)  # Pause to avoid overwhelming your crawler
