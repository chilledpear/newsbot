import logging
import json
import os
from datetime import datetime

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/newsbot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Set up logging with a file and console handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

class DebugLogger:
    """A class to handle all debugging output for the news bot."""
    
    @staticmethod
    def log_search_term(user_topic, exclusion_terms):
        """Log the user's search term and exclusion terms."""
        logging.info(f"🔍 SEARCH QUERY: User topic: '{user_topic}', Exclusion terms: '{exclusion_terms}'")
    
    @staticmethod
    def log_keywords(keywords):
        """Log the keywords generated from the search term."""
        logging.info(f"🔤 KEYWORDS GENERATED: {json.dumps(keywords)}")
    
    @staticmethod
    def log_duckduckgo_results(phrase, results_count):
        """Log the results from DuckDuckGo search."""
        logging.info(f"🦆 DUCKDUCKGO SEARCH: Phrase '{phrase}' returned {results_count} results")
    
    @staticmethod
    def log_duckduckgo_article(title, url, date):
        """Log details of an article found from DuckDuckGo."""
        logging.info(f"📰 ARTICLE FOUND: Title: '{title}', URL: {url}, Date: {date}")
    
    @staticmethod
    def log_prefilter_decision(title, is_kept, reason):
        """Log the decision from the first filtering step."""
        status = "KEPT" if is_kept else "REJECTED"
        logging.info(f"🔍 FIRST FILTER {status}: '{title}' - Reason: {reason}")
    
    @staticmethod
    def log_batch_filter_started(count):
        """Log the start of batch filtering."""
        logging.info(f"⚙️ BATCH FILTER STARTED: Processing {count} articles")
    
    @staticmethod
    def log_batch_filter_completed(kept, total):
        """Log the completion of batch filtering."""
        logging.info(f"⚙️ BATCH FILTER COMPLETED: Kept {kept}/{total} articles")
    
    @staticmethod
    def log_detailed_filtering(title, url, checks):
        """Log detailed information about article filtering."""
        logging.info(f"🔬 DETAILED ANALYSIS: '{title}' - {url}")
        for check_name, result, detail in checks:
            status = "PASSED" if result else "FAILED"
            logging.info(f"  • {check_name}: {status} - {detail}")
    
    @staticmethod
    def log_final_decision(title, is_accepted, reason, quality_score=None):
        """Log the final decision on an article."""
        status = "ACCEPTED" if is_accepted else "REJECTED"
        score_info = f", Quality Score: {quality_score}" if quality_score is not None else ""
        logging.info(f"✅ FINAL {status}: '{title}' - Reason: {reason}{score_info}")
    
    @staticmethod
    def log_search_summary(total_found, total_kept):
        """Log summary of the search results."""
        logging.info(f"📊 SEARCH SUMMARY: Found {total_found} articles, kept {total_kept} after filtering")

# Replace existing debug logger with our new implementation
DEBUG = DebugLogger()