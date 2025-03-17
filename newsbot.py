import asyncio
import hashlib
import json
import os
import re
import random
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Callable, Any, Union
from functools import wraps

import aiohttp
import openai
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes,
    MessageHandler, filters
)

# Load environment variables
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or not BOT_TOKEN:
    raise ValueError("Missing API keys. Set OPENAI_API_KEY and BOT_TOKEN in .env file.")

openai.api_key = OPENAI_API_KEY

# Global settings
MSN_ENABLED = False  # Changed to False to skip MSN articles by default
CACHE = {}
MAX_CACHE_SIZE = 1000
SEARCH_DELAY = 1
SEARCH_INTERVAL = 10
RETRY_DELAY = 1
SEND_DELAY = 2
MAX_SEND_ATTEMPTS = 3
DEBUG_MODE = True
SHUTDOWN_FLAG = False  # Flag to signal graceful shutdown
APP_INSTANCE = None  # Store the application instance globally
USED_KEYWORDS = {}  # Dictionary to track used keywords by topic, keyed by topic

# Track domains that are consistently access-restricted
ACCESS_RESTRICTED_DOMAINS = {}

# Create a global set to track all processed URLs
PROCESSED_URLS = set()

# User agents
DESKTOP_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
MOBILE_USER_AGENT = "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"

# Article tracking
ARTICLE_STATS = {
    "total_discovered": 0, "fetch_failed": 0, "evaluation_failed": 0,
    "low_relevance": 0, "old_news": 0, "approved": 0, "sent": 0,
    "articles": []
}

# Store processed articles for reference
ARTICLE_HISTORY = []
MAX_HISTORY_SIZE = 100

# Custom emoji that display properly
EMOJI = {
    "approved": "✅", "rejected": "❌", "duplicate": "🔄",
    "fetch_failed": "📵", "eval_failed": "⛔", "old_news": "🕒",
    "sent": "📨", "unknown": "❓", "debug": "🔍"
}

###########################################
# Core Utility Functions
###########################################

def extract_domain(url):
    """Extract domain from URL"""
    match = re.search(r'https?://([^/]+)', url)
    if match:
        return match.group(1).lower()
    return ""

def debug_log(message):
    """Log debug message if debug mode is enabled"""
    if DEBUG_MODE:
        print(f"{EMOJI['debug']} DEBUG: {message}")

def print_article_stats():
    """Print comprehensive article processing statistics"""
    print("\n🔢 ======================= SUMMARY STATS ======================= 🔢")
    print(f"📊 Total discovered:  {ARTICLE_STATS['total_discovered']}")
    print(f"📵 Failed to fetch:   {ARTICLE_STATS['fetch_failed']}")
    print(f"⛔ Failed evaluation: {ARTICLE_STATS['evaluation_failed']}")
    print(f"❌ Low relevance:     {ARTICLE_STATS['low_relevance']}")
    print(f"🕒 Old news:          {ARTICLE_STATS['old_news']}")
    print(f"✅ Approved:          {ARTICLE_STATS['approved']}")
    print(f"📨 Actually sent:     {ARTICLE_STATS['sent']}")
    
    print("\n📰 ==================== ARTICLE BREAKDOWN ==================== 📰")
    for article in ARTICLE_STATS['articles'][-20:]:
        status = article['status']
        icon = EMOJI.get(status.lower(), EMOJI["unknown"])
        print(f"{icon} [{status}] {article['title'][:50]}... - {article.get('reason', '')}")

def get_cache_key(text: str) -> str:
    """Create cache key from text"""
    return hashlib.md5(text.encode()).hexdigest()

def cached(func):
    """Decorator for caching function results"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Create a cache key based on function name and arguments
        key_parts = [func.__name__]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in kwargs.items()])
        cache_key = get_cache_key(":".join(key_parts))
        
        # Check if result is in cache
        if cache_key in CACHE:
            return CACHE[cache_key]
        
        # Call the original function
        result = await func(*args, **kwargs)
        
        # Store result in cache
        if len(CACHE) >= MAX_CACHE_SIZE:
            # Remove 20% of oldest items
            keys_to_remove = random.sample(list(CACHE.keys()), int(MAX_CACHE_SIZE * 0.2))
            for k in keys_to_remove:
                CACHE.pop(k, None)
        
        CACHE[cache_key] = result
        return result
    return wrapper

def handle_errors(func):
    """Decorator for handling errors in async functions"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            function_name = func.__name__
            print(f"⚠️ Error in {function_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return appropriate default values based on function name
            if "search" in function_name:
                return []
            elif "evaluate" in function_name:
                return None
            elif "fetch" in function_name:
                return None
            else:
                return None
    return wrapper

def add_to_article_history(article):
    """Add article to history for duplicate detection"""
    global ARTICLE_HISTORY
    
    ARTICLE_HISTORY.append({
        "title": article.get("title", ""),
        "url": article.get("url", ""),
        "content_snippet": article.get("content", "")[:500] if article.get("content") else "",
        "source": article.get("source", ""),
        "processed_at": datetime.now()
    })
    
    # Keep only the latest MAX_HISTORY_SIZE articles
    if len(ARTICLE_HISTORY) > MAX_HISTORY_SIZE:
        ARTICLE_HISTORY = ARTICLE_HISTORY[-MAX_HISTORY_SIZE:]

def parse_json(text: str) -> dict:
    """Extract and parse JSON from text with improved error handling for markdown code blocks"""
    try:
        # Clean any markdown code block markers
        clean_text = re.sub(r'```(json|javascript|python)?\s*', '', text)
        clean_text = re.sub(r'```\s*$', '', clean_text)
        
        # First try direct parsing
        return json.loads(clean_text)
    except json.JSONDecodeError:
        # Try to extract JSON from the text
        try:
            # Find patterns that look like JSON objects
            json_pattern = re.compile(r'(\{.*\})', re.DOTALL)
            match = json_pattern.search(clean_text)
            if match:
                return json.loads(match.group(1))
            
            # If no match, try array pattern
            json_array_pattern = re.compile(r'(\[.*\])', re.DOTALL)
            match = json_array_pattern.search(clean_text)
            if match:
                return json.loads(match.group(1))
        except Exception as e:
            print(f"Failed to extract JSON: {e}")
        
        print(f"Could not parse JSON from: {clean_text[:100]}...")
        return {}

def write_approved_articles_to_file(filename="approved_articles.txt"):
    """
    Write all approved articles to a text file for manual review.
    This helps identify false positives in the article approval process.
    
    Args:
        filename (str): The name of the output file
    """
    approved_articles = [article for article in ARTICLE_STATS['articles'] 
                        if article.get('status') == 'APPROVED']
    
    if not approved_articles:
        print(f"No approved articles to write to {filename}")
        return
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Approved Articles Analysis\n\n")
        f.write("This file contains details of all articles that were approved by the news bot's evaluation system.\n")
        f.write("Use this to identify and correct false positives in the article selection process.\n\n")
        
        f.write("## Approved Articles\n\n")
        for i, article in enumerate(approved_articles, 1):
            f.write(f"### Article {i}\n")
            f.write(f"**Title**: {article.get('title', '')}\n")
            f.write(f"**Source**: {article.get('source', '')}\n")
            f.write(f"**Date**: {article.get('date', '')}\n")
            f.write(f"**URL**: {article.get('url', '')}\n")
            f.write(f"**Search Query**: {article.get('query', '')}\n\n")
            
            f.write("**Viral Metrics**:\n")
            f.write(f"- Query Match Score: {article.get('query_match', 'N/A')}/5\n")
            f.write(f"- Specificness Score: {article.get('specificness', 'N/A')}/5\n")
            f.write(f"- Substantiveness: {article.get('substantiveness', 'N/A')}/5\n")
            f.write(f"- Uniqueness: {article.get('uniqueness', 'N/A')}/5\n")
            f.write(f"- Front Page Score: {article.get('score', 'N/A')}/5\n")
            f.write(f"- Viral Potential: {article.get('viral_potential', 'N/A')}/5\n\n")
            
            f.write(f"**Approval Reasoning**: {article.get('reason', '')}\n\n")
            f.write("---\n\n")
            
    print(f"✅ Successfully wrote {len(approved_articles)} approved articles to {filename}")

###########################################
# API Interaction Functions
###########################################

async def ask_openai(prompt: str, system_role: str = "", temperature: float = 0.5) -> str:
    """Improved OpenAI API call with better error handling and retry logic"""
    messages = []
    if system_role:
        messages.append({"role": "system", "content": system_role})
    messages.append({"role": "user", "content": prompt})
    
    # Exponential backoff parameters
    max_retries = 5
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            # Use concurrent.futures to run the API call in a thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=temperature,
                    request_timeout=30
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API Error: {str(e)} (attempt {attempt+1}/{max_retries})")
        
        # Sleep with exponential backoff before retrying
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            print(f"Retrying in {delay:.2f} seconds...")
            await asyncio.sleep(delay)
    
    # If we've exhausted all retries, return a default response
    print("All OpenAI API attempts failed. Using fallback.")
    return "OPENAI_ERROR"

@cached
@handle_errors
async def search_news(query: str, max_results: int = 30, timelimit: str = "d") -> List[Dict]:
    """Search for news with caching and rate limiting protection"""
    query = query.strip()
    
    async def perform_search(query_text, attempt=0):
        try:
            debug_log(f"Sending search query: '{query_text}'")
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    query_text, region="wt-wt", safesearch="Off", 
                    timelimit=timelimit, max_results=max_results
                ))
                
            debug_log(f"Received {len(results)} raw results from search")
            
            articles = []
            for result in results:
                if not result.get("title") or not result.get("url"):
                    continue
                
                url = result.get("url", "").lower()
                
                # Skip MSN articles (MSN_ENABLED is False by default now)
                if "msn.com" in url and not MSN_ENABLED:
                    continue
                
                # Skip domains that are known to be access-restricted
                domain = extract_domain(url)
                if domain in ACCESS_RESTRICTED_DOMAINS:
                    debug_log(f"Skipping access-restricted domain: {domain}")
                    continue
                    
                articles.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "description": result.get("body", ""),
                    "source": result.get("source", ""),
                    "date": result.get("date", ""),
                    "query": query_text,  # Store the original search query
                    "_id": get_cache_key(f"{result.get('url', '')}:{result.get('title', '')}")
                })
                
            return articles
        except Exception as e:
            print(f"📵 Search error: {str(e)}. Retrying...")
            if attempt < 2:
                await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                if attempt == 1:
                    simple_query = " ".join(query_text.split()[:3])
                    print(f"Trying with simplified query: '{simple_query}'")
                    return await perform_search(simple_query, attempt + 1)
                return await perform_search(query_text, attempt + 1)
            else:
                print(f"❌ All search attempts failed for '{query_text}'")
                return []
    
    print(f"\n🔎 Searching for: '{query}'")
    articles = await perform_search(query)
    
    if articles:
        print(f"Found {len(articles)} valid results for '{query}'")
        
        # Update article stats
        ARTICLE_STATS['total_discovered'] += len(articles)
        for article in articles:
            ARTICLE_STATS['articles'].append({
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'status': 'DISCOVERED',
                'query': query
            })
    else:
        print(f"No articles found for query: '{query}'")
    
    # Add delay to prevent rate limiting
    await asyncio.sleep(SEARCH_DELAY)
    
    return articles

@cached
@handle_errors
async def fetch_content(url: str) -> Optional[str]:
    """Advanced fetch function that can handle MSN and other JavaScript-heavy sites"""
    async def try_fetch(retry=False, use_mobile=False):
        try:
            # Use mobile or desktop user agent based on parameter
            user_agent = MOBILE_USER_AGENT if use_mobile else DESKTOP_USER_AGENT
            
            # Enhanced browser-like headers with full browser signature
            headers = {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Referer": "https://www.google.com/",
                "Cache-Control": "max-age=0",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "cross-site",
                "Sec-Fetch-User": "?1",
                "Upgrade-Insecure-Requests": "1",
                "DNT": "1"
            }
            
            # Setup session with browser-like behavior
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=25)  # Increased timeout
                
                # For MSN URLs, try to access the mobile version which is simpler
                request_url = url
                if "msn.com" in url and not "/amp/" in url and not use_mobile:
                    # Try to construct a mobile version URL
                    parts = url.split('.com/')
                    if len(parts) > 1:
                        request_url = parts[0] + '.com/amp/' + parts[1]
                        debug_log(f"Trying MSN mobile URL: {request_url}")
                
                async with session.get(request_url, ssl=False, timeout=timeout, headers=headers, allow_redirects=True) as resp:
                    if resp.status != 200:
                        print(f"📵 Failed to fetch {request_url} - Status code: {resp.status}")
                        # If we tried a modified URL and failed, fall back to original
                        if request_url != url:
                            debug_log(f"Falling back to original URL: {url}")
                            async with session.get(url, ssl=False, timeout=timeout, headers=headers, allow_redirects=True) as orig_resp:
                                if orig_resp.status != 200:
                                    return None
                                html = await orig_resp.text()
                        else:
                            return None
                    else:
                        html = await resp.text()
                    
                    # Check for common blocking patterns
                    if len(html) < 500 or "access denied" in html.lower() or "captcha" in html.lower():
                        print(f"📵 Access restricted for {request_url}")
                        
                        # Track access restriction for this domain
                        domain = extract_domain(request_url)
                        if domain:
                            if domain in ACCESS_RESTRICTED_DOMAINS:
                                # This is the second time this domain has been restricted
                                print(f"🚫 Domain {domain} has been access-restricted multiple times")
                            else:
                                # First time this domain is restricted
                                ACCESS_RESTRICTED_DOMAINS[domain] = datetime.now()
                                print(f"🚫 Adding {domain} to access-restricted domains list")
                                
                        return None
                        
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Debug: Count total text content size
                    all_text = soup.get_text()
                    debug_log(f"Total text size from {request_url}: {len(all_text)} chars")
                    
                    # If text is suspiciously short, do additional checks
                    if len(all_text) < 1000:
                        debug_log(f"Short content detected, trying special extraction for {request_url}")
                        
                        # For MSN: Look for JSON data that contains the article content
                        if "msn.com" in url:
                            # Look for JSON-LD or other structured data
                            json_scripts = soup.find_all("script", type="application/ld+json")
                            for script in json_scripts:
                                try:
                                    data = json.loads(script.string)
                                    # Look for articleBody in the JSON
                                    if isinstance(data, dict) and "articleBody" in data:
                                        content = data["articleBody"]
                                        if len(content) > 200:
                                            return content
                                    # Handle array type
                                    elif isinstance(data, list):
                                        for item in data:
                                            if isinstance(item, dict) and "articleBody" in item:
                                                content = item["articleBody"]
                                                if len(content) > 200:
                                                    return content
                                except:
                                    pass
                            
                            # Try to find the cached content inside JavaScript variables
                            content_pattern = re.search(r'"content":"(.*?)","description"', html)
                            if content_pattern:
                                content = content_pattern.group(1)
                                # Clean up escaped characters
                                content = content.replace('\\"', '"').replace('\\n', ' ')
                                if len(content) > 200:
                                    return content
                            
                            # MSN sometimes has a window.__data variable with content
                            json_data_pattern = re.search(r'window\.__data\s*=\s*({.*?});', html, re.DOTALL)
                            if json_data_pattern:
                                try:
                                    json_data = json.loads(json_data_pattern.group(1))
                                    # Navigate the complex JSON structure to find article content
                                    if 'article' in json_data and 'body' in json_data['article']:
                                        return json_data['article']['body']
                                    elif 'context' in json_data and 'currentArticle' in json_data['context']:
                                        article_data = json_data['context']['currentArticle']
                                        if 'body' in article_data:
                                            return article_data['body']
                                except Exception as e:
                                    debug_log(f"Error extracting MSN content from JSON: {e}")
                    
                    # Remove non-content elements
                    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
                        tag.decompose()
                    
                    # Try multiple strategies to find content
                    content = ""
                    
                    # Site-specific extraction for common sites
                    if "msn.com" in url:
                        # MSN specific selectors (expanded)
                        selectors = [
                            "div[data-testid='content-canvas']",
                            ".article-body",
                            ".articlecontent",
                            ".article-content",
                            "article .primary-content",
                            "div[data-impress='article']",
                            "div.article-content-wrapper",
                            "#maincontent",
                            "[class*='ArticleBody']",
                            "[data-t*='ArticleBody']"
                        ]
                        
                        for selector in selectors:
                            element = soup.select_one(selector)
                            if element:
                                content = element.get_text(separator=" ", strip=True)
                                if len(content) > 100:
                                    return content
                    
                    # General content extraction strategies
                    content_candidates = []
                    
                    # 1. Try article element
                    if article := soup.find("article"):
                        content = article.get_text(separator=" ", strip=True)
                        content_candidates.append((content, 10))  # Higher priority
                    
                    # 2. Try main content areas
                    for selector in ["main", "div.content", "div.article", "div.post", "#content", "#main"]:
                        element = soup.select_one(selector)
                        if element:
                            text = element.get_text(separator=" ", strip=True)
                            content_candidates.append((text, 8))
                    
                    # 3. Look for content divs by class/id patterns
                    for element in soup.find_all(["div", "section"]):
                        for attr in ["class", "id"]:
                            if element.has_attr(attr):
                                attr_value = " ".join(element[attr]) if isinstance(element[attr], list) else element[attr]
                                if any(pattern in attr_value.lower() for pattern in ["content", "article", "story", "body", "text"]):
                                    text = element.get_text(separator=" ", strip=True)
                                    priority = 6
                                    if len(text) > 1000:  # Longer content gets higher priority
                                        priority += 2
                                    content_candidates.append((text, priority))
                    
                    # 4. Try paragraphs
                    paragraphs = soup.find_all("p")
                    if paragraphs and len(paragraphs) > 2:  # Make sure we have at least a few paragraphs
                        combined_paragraphs = " ".join(p.get_text().strip() for p in paragraphs)
                        content_candidates.append((combined_paragraphs, 4))
                        
                    # 5. Last resort: Find largest text block
                    divs = soup.find_all("div")
                    longest_text = ""
                    for div in divs:
                        text = div.get_text(separator=" ", strip=True)
                        if len(text) > len(longest_text):
                            longest_text = text
                    content_candidates.append((longest_text, 1))  # Lowest priority
                    
                    # Sort candidates by length and priority, then pick the best
                    content_candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
                    
                    for candidate, _ in content_candidates:
                        if len(candidate) >= 200:  # More generous minimum length
                            # Clean up content
                            clean_content = re.sub(r'\s+', ' ', candidate).strip()
                            return clean_content[:10000]  # Increased limit
                    
                    # If we reach here, no good candidate was found
                    if longest_text and len(longest_text) > 100:
                        return longest_text[:10000]
                        
                    print(f"📵 Content too short for {request_url} - Could not extract meaningful content")
                    return None
                        
        except Exception as e:
            print(f"📵 Error fetching {url}: {str(e)}")
            return None
    
    # First attempt with desktop user agent
    content = await try_fetch()
    
    # Retry with mobile user agent if desktop failed
    if not content:
        await asyncio.sleep(RETRY_DELAY)
        content = await try_fetch(retry=True, use_mobile=True)
        
    # Try archive.org as a last resort for important sites
    if not content and any(site in url for site in ["msn.com", "cbssports.com", "express.co.uk"]):
        try:
            archive_url = f"https://web.archive.org/web/{url}"
            debug_log(f"Trying archive.org: {archive_url}")
            archive_content = await try_fetch_with_url(archive_url)
            if archive_content and len(archive_content) > 200:
                content = archive_content
                debug_log(f"Successfully retrieved content from archive.org")
        except Exception as e:
            debug_log(f"Archive.org fetch failed: {e}")
    
    if not content:
        # Update stats for failed fetch
        ARTICLE_STATS['fetch_failed'] += 1
        # Update status of corresponding article
        for art in ARTICLE_STATS['articles']:
            if art.get('url') == url and art.get('status') == 'DISCOVERED':
                art['status'] = 'FETCH_FAILED'
                art['reason'] = 'Failed to retrieve content'
                break
        
    return content

async def try_fetch_with_url(url):
    """Helper function to fetch content from a specific URL"""
    headers = {
        "User-Agent": DESKTOP_USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }
    
    async with aiohttp.ClientSession() as session:
        timeout = aiohttp.ClientTimeout(total=20)
        try:
            async with session.get(url, ssl=False, timeout=timeout, headers=headers, allow_redirects=True) as resp:
                if resp.status != 200:
                    return None
                    
                html = await resp.text()
                
                if len(html) < 500:
                    return None
                    
                soup = BeautifulSoup(html, "html.parser")
                
                # Remove non-content elements
                for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    tag.decompose()
                
                content = ""
                
                # Try to find main content
                if article := soup.find("article"):
                    content = article.get_text(separator=" ", strip=True)
                elif main := soup.find("main"):
                    content = main.get_text(separator=" ", strip=True)
                else:
                    # Try paragraphs as a fallback
                    paragraphs = soup.find_all("p")
                    if paragraphs:
                        content = " ".join(p.get_text().strip() for p in paragraphs[:25])
                
                # Clean and limit content
                content = re.sub(r"\s+", " ", content).strip()
                
                if len(content) < 100:
                    return None
                    
                return content[:8000]
        except Exception:
            return None

###########################################
# Consolidated News Processing Functions
###########################################

@handle_errors
async def generate_keywords_batch(topic, existing_keywords=None):
    """Generate search keywords using a single GPT call with improved frontpage understanding"""
    context = ""
    if existing_keywords:
        context = f"I've already used these keywords: {', '.join(existing_keywords)}. "
    
    current_year = datetime.now().year
    prompt = f"""{context}Generate 5 colloqial search queries for finding breakingnews about '{topic}' that would be PERFECT for the front page of a viral content site like Buzzfeed, Twitter, New York Post,or TMZ.

The queries should target content that is:
- Highly shareable and viral-worthy
- Emotional, surprising, controversial, or inspiring

Return ONLY the queries as a simple comma-separated list without numbering, quotes, or formatting."""
    
    system_role = """You are an expert at identifying only the most viral, and the most shareable content. 
    You understand what makes headlines go viral, what content performs well on social media, and you filter out everything that does not make the cut.
    You create search queries that find exactly this type of content - emotional, surprising, trendy, and shareable."""
    
    response = await ask_openai(prompt, system_role, temperature=0.7)
    debug_log(f"Raw OpenAI response for keywords: {response}")
    
    if response == "OPENAI_ERROR":
        return [f"viral {topic} news", f"shocking {topic} revealed", f"must-see {topic} trends"]
    
    # Parse keywords from the response
    keywords = []
    
    # First try comma-separated format
    if "," in response:
        keywords = [k.strip() for k in response.split(",") if k.strip()]
    else:
        # Handle numbered list or line-by-line format
        lines = [line.strip() for line in response.split('\n')]
        for line in lines:
            cleaned = re.sub(r'^\d+[\.\)]\s*|^[-•*]\s*|[""\']', '', line).strip()
            if cleaned:
                keywords.append(cleaned)
    
    # Filter out any empty strings and ensure no duplicates
    keywords = [k for k in keywords if k]
    
    # Ensure we have at least one keyword
    if not keywords:
        keywords = [f"viral {topic} trending", f"shocking {topic} news"]
    
    debug_log(f"Processed keywords: {keywords}")
    return keywords

@handle_errors
async def evaluate_articles_batch(articles, topic, query):
    """Process multiple articles with a focus on front-page, viral-worthy content and query matching"""
    if not articles:
        return []
    
    # Prepare minimal article data to save tokens
    articles_data = []
    for i, article in enumerate(articles):
        # Get content or description snippet
        content_snippet = ""
        if article.get("content"):
            content_snippet = article.get("content")[:1000]
        elif article.get("description"):
            content_snippet = article.get("description")
        
        articles_data.append({
            "id": i,
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "date": article.get("date", ""),
            "snippet": content_snippet[:200] if content_snippet else "",
        })
    
    # Build prompt for batch evaluation with critical stance and focus on query matching and specificness
    prompt = f"""You are a SELECTIVE content critic evaluating news articles about "{topic}" that match the search query: "{query}".

START by carefully evaluating how well each article matches the SPECIFIC search query: "{query}".

ARTICLES:
{json.dumps(articles_data, indent=2)}

For each article, evaluate with a critical eye:

1. QUERY MATCH (1-5 scale): Does this PRECISELY address the specific search query "{query}"? This is the top priority.

2. SPECIFICNESS (1-5 scale): Does the article mention SPECIFIC names, identifiers, or details? This is the second priority.
   - High scoring (4-5): Contains exact names of people, places, specific products, organizations, or clearly named events
   - Low scoring (1-2): Uses vague terms like "scientists", "researchers", "experts" without specific identities

3. SUBSTANTIVENESS (1-5 scale): Does this contain meaningful, substantial content beyond clickbait?

4. UNIQUENESS (1-5 scale): Is this truly novel information not widely reported elsewhere?

5. FRONT-PAGE QUALITY (1-5 scale): Would a quality publication put this on their front page?

6. VIRAL POTENTIAL (1-5 scale): Would people actually share this with others?

REJECT if the article:
- Is not directly addressing the specific search query "{query}"
- Lacks specific named entities, people, products, events, etc. (too vague or general)
- Is simply aggregating information available elsewhere
- Has a misleading or over-sensationalized headline
- Lacks authoritative sources or substantive content
- Would not truly surprise or inform the reader

Return ONLY a JSON array without any markdown formatting, with each entry containing:
- id: The article ID
- query_match: 1-5 rating (be strict on this!)
- specificness: 1-5 rating (be strict - does it name specific entities/people/events?)
- substantiveness: 1-5 rating
- uniqueness: 1-5 rating
- front_page_score: 1-5 rating
- viral_potential: 1-5 rating
- show_article: true/false (default to false)
- reason: Explain specifically WHY this should be rejected OR approved (be specific)
- is_old_news: true/false

Only mark show_article as true if these criteria are met:
1. query_match is at least 4
2. specificness is at least 3
3. substantiveness is at least 3
4. uniqueness is at least 3
5. front_page_score is at least 3
6. viral_potential is at least 3

Be selective - it's better to approve quality articles that truly match the query than to approve everything.
"""

    system_role = """You are a selective news critic with high standards.

You carefully evaluate content for:
- Relevance to the specific search query (highest priority)
- Specificness - containing named entities, specific people, events, products, etc. (2nd highest priority)
- Original reporting with unique insights
- Substantive content that truly informs
- Quality that would merit prominent placement in good publications
- Genuine viral potential based on substance, not just clickbait

You actively filter out:
- Content that doesn't directly address the search query
- Vague articles that don't mention specific names or identifiable entities
- Aggregated content that simply recycles information
- Clickbait or overblown headlines
- Content that merely repackages existing knowledge

Your default position is critical evaluation. Articles must demonstrate quality, specificity and relevance to earn your approval."""
    
    response = await ask_openai(prompt, system_role, temperature=0.3)
    
    if response == "OPENAI_ERROR":
        # Simple fallback evaluation
        default_approved = []
        for i, article in enumerate(articles[:3]):  # Limit to top 3 in fallback mode
            default_approved.append({
                "id": i,
                "query_match": 3,
                "specificness": 3,
                "substantiveness": 3,
                "uniqueness": 3,
                "front_page_score": 3,
                "viral_potential": 3,
                "show_article": True,
                "reason": f"Article about {topic} that could go viral (fallback evaluation)",
                "is_old_news": False
            })
        return process_evaluation_results(default_approved, articles)
        
    # Parse the results
    result = parse_json(response)
    if not isinstance(result, list):
        print(f"⛔ Invalid evaluation result format: {response[:100]}...")
        return []
    
    return process_evaluation_results(result, articles)

def process_evaluation_results(evaluation_results, articles):
    """Process evaluation results and update article statuses with viral/front-page focus"""
    approved_articles = []
    
    for result in evaluation_results:
        try:
            article_id = result.get("id")
            if article_id is None or article_id >= len(articles):
                continue
                
            article = articles[article_id]
            
            # Get evaluation details
            query_match = result.get("query_match", 0)
            specificness = result.get("specificness", 0)
            substantiveness = result.get("substantiveness", 0)
            uniqueness = result.get("uniqueness", 0)
            front_page_score = result.get("front_page_score", 0)
            viral_potential = result.get("viral_potential", 0)
            show_article = result.get("show_article", False)
            reason = result.get("reason", "No reason provided")
            is_old_news = result.get("is_old_news", False)
            
            # Convert string to boolean if needed
            if isinstance(show_article, str):
                show_article = show_article.lower() == "true"
            if isinstance(is_old_news, str):
                is_old_news = is_old_news.lower() == "true"
                
            # Add analysis to article
            article["analysis"] = result
            
            # Create a detailed score display
            score_display = f"[QM:{query_match}/5 | SP:{specificness}/5 | S:{substantiveness}/5 | U:{uniqueness}/5 | FP:{front_page_score}/5 | V:{viral_potential}/5]"
            
            # Update article status based on evaluation
            if is_old_news:
                # Old news
                ARTICLE_STATS['old_news'] += 1
                print(f"{EMOJI['old_news']} OLD NEWS: {article.get('title', '')[:70]}... - {reason}")
                
                # Update article status
                for art in ARTICLE_STATS['articles']:
                    if art.get('title') == article.get('title', '') and art.get('status') == 'DISCOVERED':
                        art['status'] = 'OLD_NEWS'
                        art['reason'] = reason
                        break
            elif not show_article or query_match < 4 or specificness < 3:
                # Low query match, specificness, viral potential or front-page quality
                ARTICLE_STATS['low_relevance'] += 1
                print(f"{EMOJI['rejected']} LOW RELEVANCE/QUALITY: {article.get('title', '')[:70]}... {score_display} - {reason}")
                
                # Update article status
                for art in ARTICLE_STATS['articles']:
                    if art.get('title') == article.get('title', '') and art.get('status') == 'DISCOVERED':
                        art['status'] = 'LOW_RELEVANCE'
                        art['score'] = front_page_score
                        art['query_match'] = query_match
                        art['specificness'] = specificness
                        art['substantiveness'] = substantiveness
                        art['uniqueness'] = uniqueness
                        art['reason'] = reason
                        break
            else:
                # Skip if URL is already in the processed set
                url = article.get("url", "")
                if url in PROCESSED_URLS:
                    print(f"{EMOJI['duplicate']} DUPLICATE URL: {article.get('title', '')[:70]}...")
                    continue
                
                # Add URL to processed set
                PROCESSED_URLS.add(url)
                
                # Approved article
                ARTICLE_STATS['approved'] += 1
                print(f"{EMOJI['approved']} APPROVED: {article.get('title', '')[:70]}... {score_display} - {reason}")
                
                # Update article status
                for art in ARTICLE_STATS['articles']:
                    if art.get('title') == article.get('title', '') and art.get('status') == 'DISCOVERED':
                        art['status'] = 'APPROVED'
                        art['score'] = front_page_score
                        art['query_match'] = query_match
                        art['specificness'] = specificness
                        art['substantiveness'] = substantiveness
                        art['uniqueness'] = uniqueness
                        art['viral_potential'] = viral_potential
                        art['reason'] = reason
                        break
                
                # Add to article history and approved list
                add_to_article_history(article)
                approved_articles.append(article)
                
            print(f"{'─' * 70}")
                
        except Exception as e:
            print(f"⚠️ Error processing evaluation result: {e}")
            continue
    
    return approved_articles

@handle_errors
async def process_search_results(search_results, topic):
    """Process search results in batches"""
    if not search_results:
        return []
    
    print(f"⚙️  Processing {len(search_results)} articles for topic: {topic}")
    
    # Filter out duplicates by URL
    unique_urls = set()
    unique_articles = []
    
    for article in search_results:
        url = article.get("url", "")
        if url and url not in unique_urls:
            unique_urls.add(url)
            unique_articles.append(article)
    
    print(f"Found {len(unique_articles)} unique articles after URL deduplication")
    
    # Filter out global duplicates by URL before content fetching
    filtered_articles = []
    for article in unique_articles:
        url = article.get("url", "")
        if url and url not in PROCESSED_URLS:
            filtered_articles.append(article)
    
    print(f"Found {len(filtered_articles)} articles after filtering already processed URLs")
    
    # Fetch content for articles in batches
    enriched_articles = []
    
    # Process in batches of 5 to limit concurrency
    batch_size = 5
    for i in range(0, len(filtered_articles), batch_size):
        batch = filtered_articles[i:i+batch_size]
        fetch_tasks = []
        
        for article in batch:
            async def fetch_and_enrich(art):
                content = await fetch_content(art["url"])
                if content:
                    art["content"] = content
                return art
                
            fetch_tasks.append(fetch_and_enrich(article))
        
        batch_results = await asyncio.gather(*fetch_tasks)
        enriched_articles.extend([a for a in batch_results if a.get("content")])
    
    # Evaluate articles in batches
    all_approved = []
    
    # Process in reasonable sized batches for evaluation
    eval_batch_size = 10
    for i in range(0, len(enriched_articles), eval_batch_size):
        batch = enriched_articles[i:i+eval_batch_size]
        
        # Group articles by their original search query
        query_groups = {}
        for article in batch:
            query = article.get("query", topic)  # Default to topic if query not available
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append(article)
        
        # Evaluate each query group separately
        for query, articles_group in query_groups.items():
            approved_batch = await evaluate_articles_batch(articles_group, topic, query)
            all_approved.extend(approved_batch)
    
    return all_approved

@handle_errors
async def find_and_send_news(topic: str, exclusions: list = None, 
                             send_callback: Callable = None, sent_ids: Set[str] = None) -> List[Dict]:
    """Find and send news with optimized processing"""
    global USED_KEYWORDS
    
    # Initialize sent_ids if not provided
    if sent_ids is None:
        sent_ids = set()
        
    # Initialize used keywords for this topic if not already done
    if topic not in USED_KEYWORDS:
        USED_KEYWORDS[topic] = []
    
    # Generate search keywords, passing previously used keywords
    keywords = await generate_keywords_batch(topic, USED_KEYWORDS[topic])
    
    # Store the newly generated keywords
    USED_KEYWORDS[topic].extend(keywords)
    
    # Keep only the last 20 keywords to avoid unbounded growth
    max_keywords_to_store = 20
    if len(USED_KEYWORDS[topic]) > max_keywords_to_store:
        USED_KEYWORDS[topic] = USED_KEYWORDS[topic][-max_keywords_to_store:]
    
    print(f"\n🔍 ======== SEARCHING for '{topic}' ======== 🔍\n")
    
    print("\n🔤 ======== KEYWORDS ======== 🔤")
    for i, kw in enumerate(keywords):
        print(f"  {i+1}. {kw}")
        
    # Filter out exclusions if needed
    if exclusions:
        keywords = [kw for kw in keywords if not any(e.lower() in kw.lower() for e in exclusions if e)]
    
    # Search for articles
    all_search_results = []
    
    for keyword in keywords:
        # Check if shutdown was requested
        if SHUTDOWN_FLAG:
            print("Shutdown requested, stopping search")
            break
            
        # Clean the keyword
        clean_keyword = keyword.strip('"\' ')
        results = await search_news(clean_keyword)
        all_search_results.extend(results)
        
        # Check if we have enough results
        if len(all_search_results) >= 50:
            break
    
    # Process search results
    approved_articles = await process_search_results(all_search_results, topic)
    
    # Send approved articles
    sent_count = 0
    for article in approved_articles:
        # Check if shutdown was requested
        if SHUTDOWN_FLAG:
            print("Shutdown requested, stopping article sending")
            break
            
        # Skip if already sent
        if article["_id"] in sent_ids:
            print(f"{EMOJI['duplicate']} ALREADY SENT: {article.get('title', '')[:70]}...")
            continue
            
        # Send the article
        if send_callback:
            try:
                # Add to sent_ids BEFORE sending to prevent duplicates
                sent_ids.add(article["_id"])
                
                # Rate limit sending
                await asyncio.sleep(SEND_DELAY)
                
                # Attempt to send
                success = await send_callback(article)
                
                if success:
                    sent_count += 1
                    ARTICLE_STATS['sent'] += 1
                    # Update article status
                    for art in ARTICLE_STATS['articles']:
                        if art.get('title') == article.get('title', '') and art.get('status') == 'APPROVED':
                            art['status'] = 'SENT'
                            break
                else:
                    print(f"⚠️ Failed to send article: {article.get('title', '')[:50]}...")
            except Exception as e:
                print(f"⚠️ Error sending article: {article.get('title', '')[:50]}... - {str(e)}")
    
    print(f"\n{'═' * 70}")
    print(f"📊 Total unique articles found: {len(all_search_results)}")
    print(f"📨 Total articles approved: {len(approved_articles)}")
    print(f"📬 Total articles sent: {sent_count}")
    print(f"{'═' * 70}\n")
    
    return approved_articles

###########################################
# Telegram Bot Handlers
###########################################

async def create_article_buttons(article):
    """Create inline keyboard buttons for an article"""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("⭐ Favorite", callback_data=f"favorite:{article['_id']}"),
        InlineKeyboardButton("🗑️ Trash", callback_data=f"trash:{article['_id']}"),
        InlineKeyboardButton("📰 Open News", url=article['url'])
    ]])

@handle_errors
async def send_article_to_user(article: Dict, chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    """Send an article to a user with enhanced front-page qualities explanation"""
    try:
        # Format the message text
        text = f"*{article['title']}*\n\n"
        
        # Add viral metrics if available
        if 'analysis' in article and article['analysis']:
            analysis = article['analysis']
            if all(k in analysis for k in ['query_match', 'specificness', 'substantiveness', 'uniqueness', 'front_page_score', 'viral_potential']):
                text += f"🎯 *Query Match:* {analysis['query_match']}/5\n"
                text += f"🔍 *Specificness:* {analysis['specificness']}/5\n"
                text += f"📚 *Substantiveness:* {analysis['substantiveness']}/5\n"
                text += f"⭐ *Uniqueness:* {analysis['uniqueness']}/5\n"
                text += f"📊 *Front Page Score:* {analysis['front_page_score']}/5\n"
                text += f"🚀 *Viral Potential:* {analysis['viral_potential']}/5\n\n"
        
        # Add explanation/summary if available
        if 'analysis' in article and article['analysis'] and 'reason' in article['analysis']:
            reason = article['analysis']['reason']
            if reason:
                text += f"{reason}\n\n"
        
        # Add link with text "link" for preview
        text += f"[link]({article['url']})"
        
        # Create reply markup
        reply_markup = await create_article_buttons(article)
        
        # Send message with retries
        max_retries = MAX_SEND_ATTEMPTS
        for attempt in range(max_retries):
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_markup=reply_markup,
                    parse_mode="Markdown",
                    disable_web_page_preview=False
                )
                return True
            except Exception as e:
                if attempt < max_retries - 1:
                    if hasattr(e, 'retry_after'):
                        await asyncio.sleep(e.retry_after + 1)
                    else:
                        await asyncio.sleep(2 * (attempt + 1))
                else:
                    # Try without markdown on final attempt
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            text=text.replace('*', ''),
                            reply_markup=reply_markup,
                            parse_mode=None,
                            disable_web_page_preview=False
                        )
                        return True
                    except:
                        return False
        
        return False
    except Exception as e:
        print(f"Error in send_article_to_user: {str(e)}")
        return False

async def get_url_from_message(message):
    """Extract URL from the message's inline keyboard if available"""
    try:
        if message and message.reply_markup and message.reply_markup.inline_keyboard:
            for row in message.reply_markup.inline_keyboard:
                for button in row:
                    if button.url:
                        return button.url
        
        # If we couldn't find the URL in the buttons, try to extract from text
        if message and message.text:
            urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', message.text)
            if urls:
                return urls[0]
        
        return "https://news.google.com"
    except Exception as e:
        print(f"Error extracting URL: {e}")
        return "https://news.google.com"

async def feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks for article feedback"""
    query = update.callback_query
    await query.answer()

    try:
        # Parse callback data
        data_parts = query.data.split(":")
        action = data_parts[0]
        article_id = data_parts[1] if len(data_parts) > 1 else None
        
        # Initialize favorites if needed
        if "favorites" not in context.chat_data:
            context.chat_data["favorites"] = {}
            
        # Handle favorite action
        if action == "favorite":
            if article_id in context.chat_data["favorites"]:
                # Remove from favorites
                del context.chat_data["favorites"][article_id]
                
                # Update keyboard
                keyboard = [[
                    InlineKeyboardButton("⭐ Favorite", callback_data=f"favorite:{article_id}"),
                    InlineKeyboardButton("🗑️ Trash", callback_data=f"trash:{article_id}"),
                    InlineKeyboardButton("📰 Open News", url=await get_url_from_message(query.message))
                ]]
                
                await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
                await query.message.reply_text("Article removed from favorites.")
            else:
                # Add to favorites
                url = await get_url_from_message(query.message)
                context.chat_data["favorites"][article_id] = {
                    "text": query.message.text,
                    "date": datetime.now(),
                    "url": url
                }
                
                # Update keyboard
                keyboard = [[
                    InlineKeyboardButton("★ Favorited", callback_data=f"favorite:{article_id}"),
                    InlineKeyboardButton("🗑️ Trash", callback_data=f"trash:{article_id}"),
                    InlineKeyboardButton("📰 Open News", url=url)
                ]]
                
                await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(keyboard))
                await query.message.reply_text("Article added to favorites! Use /favorite to view your saved articles.")
        
        # Handle trash action
        elif action == "trash":
            # Remove from favorites if present
            if article_id in context.chat_data.get("favorites", {}):
                del context.chat_data["favorites"][article_id]
            
            # Edit message
            await query.message.edit_text(
                f"~~{query.message.text}~~\n\n_Article moved to trash_",
                parse_mode="Markdown"
            )
            
            # Remove reply markup
            await query.edit_message_reply_markup(reply_markup=None)
    except Exception as e:
        print(f"Error in feedback_callback: {e}")

async def search_job(context: ContextTypes.DEFAULT_TYPE):
    """Background job to search for news"""
    chat_id = context.job.chat_id
    topic = context.chat_data.get("topic", "")
    
    if not topic:
        return
    
    # Get exclusions
    exclusions = context.chat_data.get("exclusions", "").split(",") if context.chat_data.get("exclusions") else []
    excluded_topics = context.chat_data.get("excluded_topics", [])
    all_exclusions = exclusions + excluded_topics
    
    # Get set of sent articles
    sent_ids = context.chat_data.get("sent_articles", set())
    
    # Send typing action
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    
    # Define send callback
    async def send_to_chat(article):
        return await send_article_to_user(article, chat_id, context)
    
    # Check if this is the initial run
    is_initial = False
    if context.job.data and isinstance(context.job.data, dict) and context.job.data.get("is_initial", False):
        is_initial = True
        context.job.data = {}  # Remove the flag for future runs
        await context.bot.send_message(chat_id=chat_id, text=f"🔍 Searching for latest news on: *{topic}*", parse_mode="Markdown")
    
    # Find and send news
    articles = await find_and_send_news(topic, all_exclusions, send_to_chat, sent_ids)
    
    # Update the sent_articles set
    for article in articles:
        sent_ids.add(article["_id"])
    context.chat_data["sent_articles"] = sent_ids

async def stop_timeout(context: ContextTypes.DEFAULT_TYPE):
    """Automatically stop search after timeout"""
    chat_id = context.job.chat_id
    
    if "job" in context.chat_data:
        context.chat_data.pop("job").schedule_removal()
        
        if "timeout_job" in context.chat_data:
            context.chat_data.pop("timeout_job").schedule_removal()
            
        await context.bot.send_message(
            chat_id=chat_id,
            text="⏰ Search automatically stopped after 30 minutes. Use /search to start a new search.",
        )

async def graceful_shutdown():
    """Perform a graceful shutdown of all async tasks with proper cleanup"""
    global SHUTDOWN_FLAG
    SHUTDOWN_FLAG = True
    
    # Give ongoing tasks some time to complete
    print("Initiating graceful shutdown...")
    await asyncio.sleep(2)
    
    try:
        # Use the application's proper shutdown method if available
        global APP_INSTANCE
        if APP_INSTANCE:
            print("Using application's built-in shutdown method...")
            try:
                # First stop the updater (if running)
                if hasattr(APP_INSTANCE, 'updater') and APP_INSTANCE.updater.running:
                    print("Stopping updater...")
                    await APP_INSTANCE.updater.stop()
                    
                # Then stop the application
                if APP_INSTANCE.running:
                    print("Stopping application...")
                    await APP_INSTANCE.stop()
                
                # Finally, shutdown the application
                print("Shutting down application...")
                await APP_INSTANCE.shutdown()
                
                print("Application shutdown completed successfully.")
                return
            except Exception as e:
                print(f"Error during application shutdown: {e}")
                print("Continuing with manual shutdown process...")
            
        # Fallback to manual task cancellation
        print("Using manual shutdown process...")
        
        # Get all running tasks except the current one
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        
        # Log number of tasks being canceled
        print(f"Canceling {len(tasks)} running tasks...")
        
        # First attempt to cancel tasks gracefully
        for task in tasks:
            task.cancel()
        
        # Wait for all tasks to be cancelled with timeout
        if tasks:
            try:
                # Set a timeout to avoid hanging indefinitely
                await asyncio.wait(tasks, timeout=5, return_when=asyncio.ALL_COMPLETED)
            except asyncio.CancelledError:
                # This is expected during cancellation
                pass
            except Exception as e:
                print(f"Error during task cancellation: {e}")
        
        print("Shutdown completed successfully.")
        
    except Exception as e:
        print(f"Error during shutdown: {e}")
    
    # Do NOT call sys.exit() directly - let the main function handle this
    # The proper way is to return from this function and let the main
    # flow terminate naturally
    print("Shutdown process complete.")

###########################################
# Command Handlers
###########################################

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    guide = (
        "Welcome to the News Bot!\n\n"
        "Commands:\n"
        "/search <topic> - Search for news on a topic\n"
        "/stop - Stop the current search\n"
        "/favorite - View your saved articles\n"
        "/setexclusions <terms> - Set terms to exclude\n"
        "/clearexclusions - Clear all exclusions\n"
        "/msntoggle - Toggle MSN results on/off\n"
        "/export - Export approved articles to file\n"
        "/help - Show this help message"
    )
    await update.message.reply_text(guide)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /search command"""
    if not context.args:
        await update.message.reply_text("Usage: /search <topic>")
        return
        
    topic = " ".join(context.args)
    
    # Send acknowledgment
    await update.message.reply_text(f"Starting search for: {topic}")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    # Store topic in context
    context.chat_data["topic"] = topic
    context.chat_data.setdefault("sent_articles", set())
    
    # Stop any existing job
    if "job" in context.chat_data:
        context.chat_data.pop("job").schedule_removal()
        
    if "timeout_job" in context.chat_data:
        context.chat_data.pop("timeout_job").schedule_removal()
    
    # Reset access-restricted domains for new search
    global ACCESS_RESTRICTED_DOMAINS
    ACCESS_RESTRICTED_DOMAINS = {}
    
    # Create a repeating job for regular updates
    job = context.job_queue.run_repeating(
        search_job,
        interval=SEARCH_INTERVAL,
        first=1,
        chat_id=update.effective_chat.id,
        data={"is_initial": True}
    )
    
    context.chat_data["job"] = job
    
    # Set a 30-minute timeout
    timeout_job = context.job_queue.run_once(
        stop_timeout,
        when=1800,
        chat_id=update.effective_chat.id
    )
    context.chat_data["timeout_job"] = timeout_job

async def stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /stop command to gracefully terminate the bot"""
    # Reset access-restricted domains
    global ACCESS_RESTRICTED_DOMAINS
    ACCESS_RESTRICTED_DOMAINS = {}
    
    # First reply to the user
    await update.message.reply_text("Stopping the news feed. Exporting approved articles...")
    
    # Cancel any scheduled jobs
    if "job" in context.chat_data:
        job = context.chat_data.pop("job")
        job.schedule_removal()
        
    if "timeout_job" in context.chat_data:
        timeout_job = context.chat_data.pop("timeout_job")
        timeout_job.schedule_removal()
    
    # Export approved articles before shutting down
    write_approved_articles_to_file()
    
    # Let the user know we're done
    await update.message.reply_text("Jobs stopped and articles exported. Type /search to start again.")
    
    # For actual bot termination, uncomment this:
    # # Set shutdown flag to prevent new operations
    # global SHUTDOWN_FLAG
    # SHUTDOWN_FLAG = True
    # # Schedule a graceful shutdown after this handler completes
    # asyncio.create_task(graceful_shutdown())

async def favorite_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /favorite command to show favorite articles"""
    favorites = context.chat_data.get("favorites", {})
    
    if not favorites:
        await update.message.reply_text("You don't have any favorite articles yet.")
        return
    
    await update.message.reply_text(f"📚 Your favorite articles ({len(favorites)}):")
    
    # Send each favorite article
    for article_id, favorite in list(favorites.items()):
        try:
            # Create buttons with the favorited state
            keyboard = [[
                InlineKeyboardButton("★ Favorited", callback_data=f"favorite:{article_id}"),
                InlineKeyboardButton("🗑️ Trash", callback_data=f"trash:{article_id}"),
                InlineKeyboardButton("📰 Open News", url=favorite.get("url", "https://news.google.com"))
            ]]
            
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=favorite["text"],
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode="Markdown",
                disable_web_page_preview=False
            )
            
            # Small delay to prevent rate limiting
            await asyncio.sleep(0.3)
        except Exception as e:
            print(f"Error sending favorite article: {e}")
            # Remove broken favorites
            favorites.pop(article_id, None)

async def setexclusions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /setexclusions command"""
    if not context.args:
        await update.message.reply_text("Usage: /setexclusions <term1,term2,...>")
        return
        
    exclusions = " ".join(context.args)
    context.chat_data["exclusions"] = exclusions
    await update.message.reply_text(f"Exclusion terms set to: {exclusions}")

async def clear_exclusions_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clearexclusions command"""
    context.chat_data["exclusions"] = ""
    context.chat_data["excluded_topics"] = []
    
    await update.message.reply_text("All exclusions cleared.")

async def msntoggle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /msntoggle command"""
    global MSN_ENABLED
    MSN_ENABLED = not MSN_ENABLED
    status = "enabled" if MSN_ENABLED else "disabled"
    await update.message.reply_text(f"MSN results {status}.")

async def debug_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /debug command to toggle debug mode"""
    global DEBUG_MODE
    DEBUG_MODE = not DEBUG_MODE
    status = "enabled" if DEBUG_MODE else "disabled"
    await update.message.reply_text(f"Debug mode {status}.")

async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /export command to export approved articles to a file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"approved_articles_{timestamp}.txt"
    write_approved_articles_to_file(filename)
    await update.message.reply_text(f"Exported approved articles to {filename}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    guide = (
        "📰 *News Bot Help Guide* 📰\n\n"
        "*Available Commands:*\n"
        "/search <topic> - Search for news on a topic\n"
        "/stop - Stop the current search\n"
        "/favorite - View your favorite articles\n"
        "/clearexclusions - Clear all exclusions\n"
        "/msntoggle - Toggle MSN results on/off\n"
        "/export - Export approved articles to file\n\n"
        "*Article Interactions:*\n"
        "⭐ Favorite - Save an article to your favorites\n"
        "🗑️ Trash - Remove an article from view\n"
        "📰 Open News - Open the original news article"
    )
    await update.message.reply_text(guide, parse_mode="Markdown")

###########################################
# Main Function
###########################################

def main():
    """Start the bot with improved shutdown handling"""
    global APP_INSTANCE
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    APP_INSTANCE = app
    
    # Add command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(CommandHandler("stop", stop_command))
    app.add_handler(CommandHandler("setexclusions", setexclusions_command))
    app.add_handler(CommandHandler("clearexclusions", clear_exclusions_command))
    app.add_handler(CommandHandler("msntoggle", msntoggle_command))
    app.add_handler(CommandHandler("favorite", favorite_command))
    app.add_handler(CommandHandler("debug", debug_command))
    app.add_handler(CommandHandler("export", export_command))
    
    # Add callback handler
    app.add_handler(CallbackQueryHandler(feedback_callback, pattern=r"^(favorite|trash):"))
    
    # Start the bot
    print("News Bot started")
    
    # Improved error handler to suppress all errors during shutdown
    async def error_handler(update, context):
        if SHUTDOWN_FLAG:
            # Suppress errors during shutdown
            return
        # For other errors, log them
        print(f"Error occurred: {context.error}")
    
    app.add_error_handler(error_handler)
    
    try:
        # Run the application
        app.run_polling(allowed_updates=Update.ALL_TYPES, close_loop=False)
    except KeyboardInterrupt:
        print("KeyboardInterrupt received, shutting down...")
        # Write approved articles before shutdown
        write_approved_articles_to_file()
        # Use run_until_complete instead of asyncio.run to avoid "Event loop is closed" errors
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in a running loop, create a task
            loop.create_task(graceful_shutdown())
        else:
            # Otherwise run the shutdown directly
            loop.run_until_complete(graceful_shutdown())
    except Exception as e:
        print(f"Error in main loop: {e}")
        # Still try to write articles even if there's an error
        write_approved_articles_to_file()
        # Same as above for graceful shutdown
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(graceful_shutdown())
            else:
                loop.run_until_complete(graceful_shutdown())
        except Exception as shutdown_error:
            print(f"Error during emergency shutdown: {shutdown_error}")
            # If all else fails, force exit
            import sys
            sys.exit(1)

    # Allow the program to exit naturally after the main function returns
    print("Main function complete, exiting normally.")

if __name__ == "__main__":
    main()