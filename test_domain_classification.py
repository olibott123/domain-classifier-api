import requests
import json

# List of domains to test
test_domains = [
    "turnkeysol.com",  # Known MSP
    "smartdolphins.com",  # Your sample domains
    "american-pcs.com",
    "control4.com",  # Residential A/V
    "savant.com"  # Commercial A/V
]

# API endpoint
API_URL = "http://localhost:5001/crawl-and-save"

def test_classification():
    results = {}
    
    for domain in test_domains:
        print(f"\nTesting domain: {domain}")
        
        try:
            response = requests.post(
                API_URL, 
                json={"url": domain},
                timeout=300  # 5-minute timeout
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Print detailed results
                print("Classification Result:")
                print(f"  Domain: {result.get('domain', 'N/A')}")
                print(f"  Company Type: {result.get('classification', {}).get('company_type', 'N/A')}")
                print(f"  Confidence Score: {result.get('classification', {}).get('confidence_score', 'N/A')}")
                print(f"  Detection Method: {result.get('classification', {}).get('detection_method', 'N/A')}")
                print(f"  Low Confidence: {result.get('classification', {}).get('low_confidence', 'N/A')}")
                
                results[domain] = result
            else:
                print(f"Error: Received status code {response.status_code}")
                results[domain] = {"error": f"Status code {response.status_code}"}
        
        except requests.exceptions.RequestException as e:
            print(f"Request error for {domain}: {e}")
            results[domain] = {"error": str(e)}
    
    # Save detailed results
    with open("classification_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nTest complete. Results saved to classification_test_results.json")

if __name__ == "__main__":
    test_classification()

