from flask import Flask, request, jsonify
from domain_classifier import DomainClassifier
from snowflake_connector import SnowflakeConnector
import requests
import time
from urllib.parse import urlparse
import json

app = Flask(__name__)

# Apify settings
APIFY_TASK_ID = "z3plE6RoQ5W6SNLDe"
APIFY_API_TOKEN = "apify_api_o4flnhGKKSc2fg25q0mUTkcojjxO4n0xiBIv"

# Initialize classifier with Pinecone
classifier = DomainClassifier(
    model_path="./domain_classifier_model_enhanced.pkl",
    use_pinecone=True,
    pinecone_api_key="pcsk_36xN79_9wLLc4wHpQEVAPTLfT2T3M7YsBrrXkqUpgPHHzuBb6iFq4Xgx4pbBdUA6VdZ67U",
    pinecone_index_name="domain-embeddings"
)

snowflake_conn = SnowflakeConnector()

def start_apify_crawl(url):
    endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
    payload = {"startUrls": [{"url": url}]}

    headers = {"Content-Type": "application/json"}

    response = requests.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()['data']['id']

def fetch_apify_results(run_id, timeout=300, interval=10):
    endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        if data:
            combined_text = ' '.join(item['text'] for item in data if 'text' in item)
            domain_url = data[0].get('url', '')
            return {
                'success': True,
                'domain': domain_url,
                'content': combined_text,
                'pages_crawled': len(data)
            }
        time.sleep(interval)

    return {'success': False, 'error': 'Timeout or no data returned'}

@app.route('/crawl-and-save', methods=['POST'])
def crawl_and_save():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL or domain required"}), 400

    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = f"https://{url}"

    try:
        crawl_run_id = start_apify_crawl(url)
        crawl_results = fetch_apify_results(crawl_run_id)

        if not crawl_results['success']:
            return jsonify({"error": crawl_results.get('error', 'Unknown error')}), 500
        
        content = crawl_results['content']
        domain = urlparse(url).netloc

        classification = classifier.classify_domain(content, domain=domain)

        success_content, error_content = snowflake_conn.save_domain_content(
            domain=domain, url=url, content=content
        )

        success_class, error_class = snowflake_conn.save_classification(
            domain=domain,
            company_type=classification['predicted_class'],
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification['confidence_scores']),
            model_metadata=json.dumps({'model_version': '1.0'}),
            low_confidence=bool(classification['low_confidence']),
            detection_method=classification['detection_method']
        )

        return jsonify({
            "domain": domain,
            "classification": classification,
            "snowflake": {
                "content_saved": bool(success_content),
                "classification_saved": bool(success_class),
                "errors": {
                    "content_error": str(error_content) if error_content else None,
                    "classification_error": str(error_class) if error_class else None
                }
            }
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
