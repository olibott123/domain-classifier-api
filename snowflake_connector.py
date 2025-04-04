import snowflake.connector
import traceback

class SnowflakeConnector:
    def __init__(self):
        self.conn_params = {
            'user': 'url_domain_crawler_testing_user',
            'private_key_path': '/root/crawler_api/rsa_key.der',
            'account': 'DOMOTZ-MAIN',
            'warehouse': 'TESTING_WH',
            'database': 'DOMOTZ_TESTING_SOURCE',
            'schema': 'EXTERNAL_PUSH',
            'authenticator': 'snowflake_jwt',
            'session_parameters': {'QUERY_TAG': 'WebCrawlerBot'}
        }

    def load_private_key(self, path):
        with open(path, "rb") as key_file:
            return key_file.read()

    def get_connection(self):
        private_key = self.load_private_key(self.conn_params['private_key_path'])
        return snowflake.connector.connect(
            user=self.conn_params['user'],
            account=self.conn_params['account'],
            private_key=private_key,
            warehouse=self.conn_params['warehouse'],
            database=self.conn_params['database'],
            schema=self.conn_params['schema'],
            authenticator=self.conn_params['authenticator'],
            session_parameters=self.conn_params['session_parameters']
        )

    def save_domain_content(self, domain, url, content):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM DOMAIN_CONTENT WHERE domain = %s AND text_content = %s
            """, (domain, content))
            if cursor.fetchone()[0] == 0:
                cursor.execute("""
                    INSERT INTO DOMAIN_CONTENT (domain, url, text_content, crawl_date)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP())
                """, (domain, url, content))
                conn.commit()
            cursor.close()
            conn.close()
            return True, None
        except Exception as e:
            return False, traceback.format_exc()

    def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO DOMAIN_CLASSIFICATION (DOMAIN, COMPANY_TYPE, CONFIDENCE_SCORE, ALL_SCORES, LOW_CONFIDENCE, DETECTION_METHOD, MODEL_METADATA, CLASSIFICATION_DATE)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            """, (domain, company_type, confidence_score, all_scores, low_confidence, detection_method, model_metadata))
            conn.commit()
            cursor.close()
            conn.close()
            return True, None
        except Exception as e:
            return False, traceback.format_exc()
