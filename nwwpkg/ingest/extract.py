"""
HTML extraction and article processing module.
"""

import json
import os
import re
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import yaml
from readability import Document

class Extractor:
    """Extract articles from URLs and HTML files using Readability."""
    
    def __init__(self, bundle_dir: str, cfg_path: str = "config/sources.yaml"):
        """
        Initialize the extractor.
        
        Args:
            bundle_dir: Path to the bundle directory
            cfg_path: Path to configuration file
        """
        self.bundle_dir = bundle_dir
        self.cfg = self._load_cfg(cfg_path)
        self.session = requests.Session()
        self._setup_session()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_cfg(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                'rate_limit': 1.0,  # seconds between requests
                'retries': 3,
                'timeout': 30
            }
    
    def _setup_session(self):
        """Setup HTTP session with configuration."""
        if 'headers' in self.cfg:
            self.session.headers.update(self.cfg['headers'])
    
    def run(self, out_path: str) -> None:
        """
        Run the extraction process.
        
        Args:
            out_path: Output path for articles.jsonl
        """
        sources = self._load_sources()
        self.logger.info(f"Processing {len(sources)} sources")
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as out:
            for i, src in enumerate(sources):
                try:
                    self.logger.info(f"Processing source {i+1}/{len(sources)}: {src}")
                    
                    # Rate limiting
                    if i > 0:
                        time.sleep(self.cfg.get('rate_limit', 1.0))
                    
                    html = self._fetch(src)
                    if html:
                        meta, text = self._readability(html, src)
                        rec = self._to_record(meta, text, src)
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out.flush()
                        
                except Exception as e:
                    self.logger.error(f"Error processing {src}: {e}")
                    continue
        
        self.logger.info(f"Extraction complete. Output saved to {out_path}")
    
    def _fetch(self, src: str) -> Optional[str]:
        """
        Fetch HTML content from source.
        
        Args:
            src: Source URL or file path
            
        Returns:
            HTML content or None if failed
        """
        if src.startswith('http'):
            return self._fetch_url(src)
        else:
            return self._fetch_file(src)
    
    def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch HTML from URL with retries."""
        for attempt in range(self.cfg.get('retries', 3)):
            try:
                response = self.session.get(
                    url, 
                    timeout=self.cfg.get('timeout', 30),
                    allow_redirects=True
                )
                response.raise_for_status()
                return response.text
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.cfg.get('retries', 3) - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"All attempts failed for {url}")
                    return None
    
    def _fetch_file(self, file_path: str) -> Optional[str]:
        """Fetch HTML from local file."""
        try:
            full_path = os.path.join(self.bundle_dir, file_path)
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def _readability(self, html: str, src: str) -> Tuple[Dict[str, Any], str]:
        """
        Extract content using Readability.
        
        Args:
            html: HTML content
            src: Source URL or file path
            
        Returns:
            Tuple of (metadata, text)
        """
        try:
            doc = Document(html)
            text = doc.summary()
            
            # Parse with BeautifulSoup for metadata
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract metadata
            meta = {
                'title': self._extract_title(soup, doc),
                'byline': self._extract_byline(soup),
                'timestamp': self._extract_timestamp(soup),
                'language': self._detect_language(text),
                'domain': self._classify_domain(text),
                'region': self._extract_region(text, src)
            }
            
            # Clean text
            text_soup = BeautifulSoup(text, 'html.parser')
            clean_text = text_soup.get_text(separator=' ', strip=True)
            
            return meta, clean_text
            
        except Exception as e:
            self.logger.error(f"Error in readability processing: {e}")
            return {}, ""
    
    def _extract_title(self, soup: BeautifulSoup, doc: Document) -> str:
        """Extract article title."""
        # Try Readability first
        if doc.title():
            return doc.title().strip()
        
        # Fallback to meta tags
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text().strip()
        
        # Try Open Graph
        og_title = soup.find('meta', property='og:title')
        if og_title:
            return og_title.get('content', '').strip()
        
        return "Untitled"
    
    def _extract_byline(self, soup: BeautifulSoup) -> str:
        """Extract article byline/author."""
        # Try common byline selectors
        byline_selectors = [
            '.byline', '.author', '.writer', '[rel="author"]',
            'meta[name="author"]', 'meta[property="article:author"]'
        ]
        
        for selector in byline_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    return element.get('content', '').strip()
                else:
                    return element.get_text().strip()
        
        return ""
    
    def _extract_timestamp(self, soup: BeautifulSoup) -> str:
        """Extract article timestamp."""
        # Try various timestamp formats
        timestamp_selectors = [
            'time[datetime]', '.timestamp', '.date', '.published',
            'meta[property="article:published_time"]',
            'meta[name="date"]'
        ]
        
        for selector in timestamp_selectors:
            element = soup.select_one(selector)
            if element:
                if element.name == 'meta':
                    timestamp = element.get('content', '')
                elif element.get('datetime'):
                    timestamp = element.get('datetime')
                else:
                    timestamp = element.get_text().strip()
                
                if timestamp:
                    return self._normalize_timestamp(timestamp)
        
        # Default to current time
        return datetime.utcnow().isoformat() + 'Z'
    
    def _normalize_timestamp(self, timestamp: str) -> str:
        """Normalize timestamp to ISO format."""
        try:
            # Try to parse and normalize
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.isoformat().replace('+00:00', 'Z')
        except:
            return datetime.utcnow().isoformat() + 'Z'
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection."""
        # Count Korean characters
        korean_chars = len(re.findall(r'[가-힣]', text))
        # Count English characters
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if korean_chars > english_chars:
            return 'ko'
        else:
            return 'en'
    
    def _classify_domain(self, text: str) -> str:
        """Classify content domain (military, diplomacy, economy)."""
        text_lower = text.lower()
        
        military_keywords = ['군사', '군대', '무기', '전쟁', '군사', 'military', 'army', 'weapon', 'war']
        diplomacy_keywords = ['외교', '정치', '협상', 'diplomacy', 'political', 'negotiation']
        economy_keywords = ['경제', '무역', '금융', 'economy', 'trade', 'finance']
        
        military_score = sum(1 for kw in military_keywords if kw in text_lower)
        diplomacy_score = sum(1 for kw in diplomacy_keywords if kw in text_lower)
        economy_score = sum(1 for kw in economy_keywords if kw in text_lower)
        
        if military_score >= max(diplomacy_score, economy_score):
            return 'military'
        elif diplomacy_score >= economy_score:
            return 'diplomacy'
        else:
            return 'economy'
    
    def _extract_region(self, text: str, src: str) -> str:
        """Extract region information."""
        # Simple region extraction based on common patterns
        region_patterns = {
            'asia': ['아시아', 'asia', '동아시아', 'east asia'],
            'europe': ['유럽', 'europe', 'eu'],
            'americas': ['미국', 'america', '북미', 'north america'],
            'middle_east': ['중동', 'middle east', 'gulf'],
            'africa': ['아프리카', 'africa']
        }
        
        text_lower = text.lower()
        for region, keywords in region_patterns.items():
            if any(kw in text_lower for kw in keywords):
                return region
        
        return 'unknown'
    
    def _to_record(self, meta: Dict[str, Any], text: str, src: str) -> Dict[str, Any]:
        """
        Convert metadata and text to article record.
        
        Args:
            meta: Extracted metadata
            text: Cleaned text content
            src: Source URL or file path
            
        Returns:
            Article record dictionary
        """
        # Generate unique ID
        content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        article_id = f"a{content_hash}"
        
        # Determine source type
        source_type = "url" if src.startswith('http') else "file"
        
        return {
            "id": article_id,
            "ts": meta.get('timestamp', datetime.utcnow().isoformat() + 'Z'),
            "title": meta.get('title', 'Untitled'),
            "text": text,
            "domain": meta.get('domain', 'unknown'),
            "region": meta.get('region', 'unknown'),
            "source": source_type,
            "byline": meta.get('byline', ''),
            "url": src if source_type == "url" else "",
            "lang": meta.get('language', 'en')
        }
    
    def _load_sources(self) -> List[str]:
        """Load sources from various input files."""
        sources = []
        
        # Try sources.csv
        csv_path = os.path.join(self.bundle_dir, "sources.csv")
        if os.path.exists(csv_path):
            import pandas as pd
            try:
                df = pd.read_csv(csv_path)
                if 'url' in df.columns:
                    sources.extend(df['url'].dropna().tolist())
            except Exception as e:
                self.logger.warning(f"Error reading sources.csv: {e}")
        
        # Try urls.txt
        txt_path = os.path.join(self.bundle_dir, "urls.txt")
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    sources.extend([line.strip() for line in f if line.strip()])
            except Exception as e:
                self.logger.warning(f"Error reading urls.txt: {e}")
        
        # Try raw HTML files
        raw_dir = os.path.join(self.bundle_dir, "raw")
        if os.path.exists(raw_dir):
            for file in os.listdir(raw_dir):
                if file.endswith('.html'):
                    sources.append(os.path.join("raw", file))
        
        if not sources:
            self.logger.warning("No sources found. Please provide sources.csv, urls.txt, or raw/*.html files.")
        
        return sources



