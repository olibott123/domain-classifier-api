from flask import Flask, request, jsonify
from flask_cors import CORS  
from domain_classifier import DomainClassifier
import requests
import time
from urllib.parse import urlparse
import os

app = Flask(__name__)
CORS(app, resources={r"/classify-domain": {"origins": "*"}})  # Allow all origins explicitly

# Apify settings
APIFY_TASK_ID = "z3plE6RoQ5W6SNLDe"
APIFY_API_TOKEN = "apify_api_o4flnhGKKSc2fg25q0mUTkcojjxO4n0xiBIv"

classifier = DomainClassifier(
    model_path=os.path.join(os.path.dirname(__file__), 'domain_classifier_model_enhanced.pkl'),
    use_pinecone=False
)
def start_apify_crawl(url):
    endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
    payload = {"startUrls": [{"url": url}]}
    response = requests.post(endpoint, json=payload)
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
            combined_text = ' '.join(item['text'] for item in data if item.get('text'))
            return combined_text
        time.sleep(interval)
    raise TimeoutError('Timeout fetching results')

@app.route('/classify-domain', methods=['POST'])
def classify_domain():
    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({"error": "URL or domain required"}), 400

    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = f"https://{url}"

    domain = urlparse(url).netloc

    try:
        crawl_run_id = start_apify_crawl(url)
        content = fetch_apify_results(crawl_run_id)

        classification = classifier.classify_domain(content, domain=domain)

        if classification["predicted_class"] == "Unknown":
            response = {
                "domain": domain,
                "predicted_class": "Unknown",
                "confidence_score": 0.0,
                "low_confidence": True,
                "reasoning": "Insufficient content length to classify accurately."
            }
        else:
            response = {
                "domain": domain,
                "predicted_class": classification['predicted_class'],
                "confidence_score": float(classification['max_confidence']),  # <-- float cast
                "low_confidence": bool(classification['low_confidence']),     # <-- bool cast
                "reasoning": f"Classified as {classification['predicted_class']} based on textual analysis."
            }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
