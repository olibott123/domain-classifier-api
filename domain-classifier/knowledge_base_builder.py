import pandas as pd
import os
import csv
import logging
from crawler import WebCrawler
from typing import List, Dict, Tuple, Optional
import random
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("knowledge_base_builder.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Company types
COMPANY_TYPES = [
    'Managed Service Provider',
    'Integrator - Residential A/V',
    'Integrator - Commercial A/V',
    'IT Department/Enterprise'
]

class KnowledgeBaseBuilder:
    def __init__(self, output_dir: str = "knowledge_base"):
        """
        Initialize the knowledge base builder
        
        Args:
            output_dir: Directory to store the knowledge base files
        """
        self.output_dir = output_dir
        self.crawler = WebCrawler()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def crawl_domain(self, domain: str) -> Tuple[bool, str]:
        """
        Crawl a domain and extract its content
        
        Args:
            domain: Domain to crawl
            
        Returns:
            Tuple of (success, content)
        """
        logger.info(f"Crawling domain: {domain}")
        success, content = self.crawler.fetch_content(domain)
        
        if not success:
            logger.warning(f"Failed to crawl {domain}: {content}")
        else:
            logger.info(f"Successfully crawled {domain}, got {len(content)} characters")
            
        return success, content
    
    def crawl_domains_list(self, domains_file: str) -> pd.DataFrame:
        """
        Crawl a list of domains from a CSV file
        
        The CSV should have these columns:
        - domain: The domain URL to crawl
        - company_type: The type of company (must match one of COMPANY_TYPES)
        
        Args:
            domains_file: Path to CSV file with domains to crawl
            
        Returns:
            DataFrame with crawled content
        """
        if not os.path.exists(domains_file):
            raise FileNotFoundError(f"Domains file not found: {domains_file}")
        
        # Read domains file
        domains_df = pd.read_csv(domains_file)
        required_columns = ['domain', 'company_type']
        for col in required_columns:
            if col not in domains_df.columns:
                raise ValueError(f"Required column '{col}' not found in {domains_file}")
        
        # Validate company types
        invalid_types = domains_df[~domains_df['company_type'].isin(COMPANY_TYPES)]['company_type'].unique()
        if len(invalid_types) > 0:
            raise ValueError(f"Invalid company types found: {invalid_types}. Must be one of {COMPANY_TYPES}")
        
        # Create results DataFrame
        results = []
        
        # Process each domain
        for _, row in domains_df.iterrows():
            domain = row['domain']
            company_type = row['company_type']
            
            # Crawl domain
            success, content = self.crawl_domain(domain)
            
            if success:
                results.append({
                    'domain': domain,
                    'company_type': company_type,
                    'content': content,
                    'status': 'success'
                })
            else:
                results.append({
                    'domain': domain,
                    'company_type': company_type,
                    'content': None,
                    'status': f'failed: {content}'
                })
            
            # Add a small delay to avoid overwhelming servers
            time.sleep(random.uniform(1, 3))
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Save raw results
        results_path = os.path.join(self.output_dir, "raw_crawl_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Raw crawl results saved to {results_path}")
        
        # Return successful crawls only
        return results_df[results_df['status'] == 'success']
    
    def create_knowledge_base_csv(self, 
                                crawled_data: Optional[pd.DataFrame] = None,
                                domains_file: Optional[str] = None,
                                max_per_type: int = 20) -> str:
        """
        Create a knowledge base CSV from crawled data
        
        Args:
            crawled_data: DataFrame with crawled content (if None, will use domains_file)
            domains_file: Path to CSV file with domains to crawl (if crawled_data is None)
            max_per_type: Maximum number of examples per company type
            
        Returns:
            Path to created knowledge base CSV
        """
        if crawled_data is None:
            if domains_file is None:
                raise ValueError("Either crawled_data or domains_file must be provided")
            crawled_data = self.crawl_domains_list(domains_file)
        
        # Filter only successful crawls with content
        valid_data = crawled_data[crawled_data['content'].notna()]
        
        # Balance the dataset with max_per_type examples per company type
        balanced_data = []
        for company_type in COMPANY_TYPES:
            type_data = valid_data[valid_data['company_type'] == company_type]
            
            if len(type_data) > max_per_type:
                # Randomly sample max_per_type examples
                type_data = type_data.sample(max_per_type, random_state=42)
            elif len(type_data) < max_per_type:
                logger.warning(f"Only {len(type_data)} examples for {company_type}, wanted {max_per_type}")
                
            balanced_data.append(type_data)
        
        # Combine all balanced data
        knowledge_base = pd.concat(balanced_data, ignore_index=True)
        
        # Save knowledge base to CSV
        kb_path = os.path.join(self.output_dir, "knowledge_base.csv")
        
        # Save with proper handling for large text content
        with open(kb_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow(['domain', 'company_type', 'content'])
            
            for _, row in knowledge_base.iterrows():
                writer.writerow([row['domain'], row['company_type'], row['content']])
        
        logger.info(f"Knowledge base saved to {kb_path} with {len(knowledge_base)} examples")
        logger.info(f"Distribution by company type:")
        for company_type in COMPANY_TYPES:
            count = len(knowledge_base[knowledge_base['company_type'] == company_type])
            logger.info(f"  {company_type}: {count} examples")
        
        return kb_path
    
    def generate_example_domains_csv(self, output_file: str = "example_domains.csv"):
        """
        Generate a template CSV file with example domains to crawl
        
        Args:
            output_file: Path to save the template CSV
        """
        example_domains = []
        
        # MSP examples - replace these with your actual domains when you have them
        msp_domains = [
            "kaseya.com", "connectwise.com", "n-able.com", "barracuda.com",
            "datto.com", "syncromsp.com", "atera.com", "pax8.com",
            "huntress.com", "sophos.com", "cisco.com/go/msp", "malwarebytes.com",
            "netwrix.com", "itarian.com", "ninjaone.com", "acronis.com",
            "egnyte.com", "passportal.com", "itglue.com", "manageengine.com"
        ]
        
        for domain in msp_domains:
            example_domains.append({
                'domain': domain,
                'company_type': 'Managed Service Provider'
            })
        
        # Residential A/V examples - replace these with your actual domains when you have them
        residential_domains = [
            "control4.com", "savant.com", "crestron.com", "snapav.com",
            "sonos.com", "lutron.com", "nest.com", "bose.com/en_us/professional.html",
            "speakercraft.com", "heos.com", "elanautomation.com", "legrand.us",
            "logitech.com/en-us/harmony.html", "meethue.com", "jblpro.com",
            "klipsch.com", "kefamerica.com", "yamaha.com/av", "denon.com",
            "sonance.com"
        ]
        
        for domain in residential_domains:
            example_domains.append({
                'domain': domain,
                'company_type': 'Integrator - Residential A/V'
            })
        
        # Commercial A/V examples - replace these with your actual domains when you have them
        commercial_domains = [
            "amx.com", "extron.com", "biamp.com", "qsc.com",
            "shure.com", "crestron.com/commercial", "polycom.com", "zoom.us",
            "microsoft.com/en-us/microsoft-teams", "logitech.com/video-collaboration",
            "clearone.com", "kramerav.com", "bssaudio.com", "harmanpro.com",
            "panasonic.com/business", "epson.com/commercial-projectors", "samsung.com/business",
            "lg.com/us/business", "nec-display.com", "peerless-av.com"
        ]
        
        for domain in commercial_domains:
            example_domains.append({
                'domain': domain,
                'company_type': 'Integrator - Commercial A/V'
            })
        
        # IT Department/Enterprise examples - replace these with your actual domains when you have them
        enterprise_domains = [
            "ibm.com", "microsoft.com", "oracle.com", "intel.com",
            "dell.com", "hp.com", "cisco.com", "vmware.com",
            "redhat.com", "juniper.net", "nutanix.com", "citrix.com",
            "salesforce.com", "servicenow.com", "adobe.com", "sap.com",
            "atlassian.com", "workday.com", "splunk.com", "tableau.com"
        ]
        
        for domain in enterprise_domains:
            example_domains.append({
                'domain': domain,
                'company_type': 'IT Department/Enterprise'
            })
        
        # Save to CSV
        pd.DataFrame(example_domains).to_csv(output_file, index=False)
        logger.info(f"Example domains CSV saved to {output_file}")
        
        return output_file

def main():
    # Create knowledge base builder
    builder = KnowledgeBaseBuilder()
    
    # Generate example domains CSV if it doesn't exist
    example_file = "example_domains.csv"
    if not os.path.exists(example_file):
        builder.generate_example_domains_csv(example_file)
        logger.info(f"Generated example domains file: {example_file}")
        logger.info(f"Please review and edit this file before continuing.")
        logger.info(f"Then run this script again to build your knowledge base.")
    else:
        # Build knowledge base
        kb_path = builder.create_knowledge_base_csv(domains_file=example_file)
        logger.info(f"Knowledge base created at {kb_path}")
        logger.info(f"You can now use this with your domain classifier.")

if __name__ == "__main__":
    main()
