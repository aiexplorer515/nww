"""
Text normalization and cleaning module.
"""

import json
import re
import hashlib
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import langdetect
from langdetect import detect

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class Normalizer:
    """Normalize and clean text content."""
    
    def __init__(self):
        """Initialize the normalizer."""
        self.logger = logging.getLogger(__name__)
        self.seen_hashes: Set[str] = set()
        
        # Language-specific patterns
        self.korean_patterns = {
            'numbers': re.compile(r'[0-9]+'),
            'units': re.compile(r'[0-9]+[가-힣]+'),
            'punctuation': re.compile(r'[^\w\s가-힣]')
        }
        
        self.english_patterns = {
            'numbers': re.compile(r'[0-9]+'),
            'units': re.compile(r'[0-9]+\s*[a-zA-Z]+'),
            'punctuation': re.compile(r'[^\w\s]')
        }
    
    def run(self, in_path: str, out_path: str, log_path: str) -> None:
        """
        Run the normalization process.
        
        Args:
            in_path: Input articles.jsonl path
            out_path: Output articles.norm.jsonl path
            log_path: Log file path
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        self.logger.info(f"Starting normalization: {in_path} -> {out_path}")
        
        processed_count = 0
        skipped_count = 0
        
        with open(in_path, 'r', encoding='utf-8') as infile, \
             open(out_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    article = json.loads(line.strip())
                    normalized = self._normalize_article(article)
                    
                    if normalized:
                        outfile.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                        processed_count += 1
                    else:
                        skipped_count += 1
                        self.logger.warning(f"Line {line_num}: Skipped duplicate or invalid article")
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Line {line_num}: JSON decode error - {e}")
                    skipped_count += 1
                except Exception as e:
                    self.logger.error(f"Line {line_num}: Processing error - {e}")
                    skipped_count += 1
        
        self.logger.info(f"Normalization complete. Processed: {processed_count}, Skipped: {skipped_count}")
    
    def _normalize_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize a single article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Normalized article or None if skipped
        """
        try:
            # Extract text content
            text = article.get('text', '').strip()
            if not text:
                self.logger.warning(f"Empty text for article {article.get('id', 'unknown')}")
                return None
            
            # Check for near-duplicates
            content_hash = self._compute_hash(text)
            if content_hash in self.seen_hashes:
                self.logger.info(f"Duplicate content detected for article {article.get('id', 'unknown')}")
                return None
            self.seen_hashes.add(content_hash)
            
            # Detect and finalize language
            lang_final = self._detect_language_final(text, article.get('lang', ''))
            
            # Normalize text
            norm_text = self._normalize_text(text, lang_final)
            
            # Segment text
            segments = self._segment_text(norm_text, lang_final)
            
            # Build normalized record
            normalized = {
                'id': article.get('id', ''),
                'ts': article.get('ts', ''),
                'lang_final': lang_final,
                'hash': content_hash,
                'norm_text': norm_text,
                'segments': segments
            }
            
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error normalizing article {article.get('id', 'unknown')}: {e}")
            return None
    
    def _compute_hash(self, text: str) -> str:
        """Compute content hash for duplicate detection."""
        # Normalize whitespace and case for hashing
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def _detect_language_final(self, text: str, initial_lang: str) -> str:
        """
        Finalize language detection.
        
        Args:
            text: Text content
            initial_lang: Initial language guess
            
        Returns:
            Final language code
        """
        try:
            # Use langdetect for more accurate detection
            detected = detect(text)
            
            # Validate detection
            if detected in ['ko', 'en']:
                return detected
            else:
                # Fallback to initial language or simple heuristic
                return initial_lang if initial_lang in ['ko', 'en'] else self._simple_lang_detect(text)
                
        except Exception:
            # Fallback to simple detection
            return initial_lang if initial_lang in ['ko', 'en'] else self._simple_lang_detect(text)
    
    def _simple_lang_detect(self, text: str) -> str:
        """Simple language detection based on character patterns."""
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        return 'ko' if korean_chars > english_chars else 'en'
    
    def _normalize_text(self, text: str, lang: str) -> str:
        """
        Normalize text content.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            Normalized text
        """
        # Basic cleaning
        text = self._clean_whitespace(text)
        text = self._normalize_punctuation(text)
        
        # Language-specific normalization
        if lang == 'ko':
            text = self._normalize_korean(text)
        else:
            text = self._normalize_english(text)
        
        # Final cleanup
        text = self._clean_whitespace(text)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Normalize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Normalize dashes
        text = re.sub(r'[–—]', '-', text)
        
        # Normalize ellipsis
        text = re.sub(r'\.{3,}', '...', text)
        
        return text
    
    def _normalize_korean(self, text: str) -> str:
        """Korean-specific normalization."""
        # Normalize numbers with units
        text = re.sub(r'([0-9]+)\s*([가-힣]+)', r'\1\2', text)
        
        # Fix common typos (basic examples)
        typos = {
            'ㅏㅏ': 'ㅏ',
            'ㅓㅓ': 'ㅓ',
            'ㅗㅗ': 'ㅗ',
            'ㅜㅜ': 'ㅜ'
        }
        
        for typo, correct in typos.items():
            text = text.replace(typo, correct)
        
        return text
    
    def _normalize_english(self, text: str) -> str:
        """English-specific normalization."""
        # Normalize numbers with units
        text = re.sub(r'([0-9]+)\s*([a-zA-Z]+)', r'\1 \2', text)
        
        # Fix common contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def _segment_text(self, text: str, lang: str) -> List[Dict[str, Any]]:
        """
        Segment text into sentences and words.
        
        Args:
            text: Normalized text
            lang: Language code
            
        Returns:
            List of segments with metadata
        """
        segments = []
        
        try:
            # Sentence segmentation
            if lang == 'ko':
                # Simple Korean sentence splitting
                sentences = re.split(r'[.!?]\s*', text)
            else:
                # Use NLTK for English
                sentences = sent_tokenize(text)
            
            # Clean and filter sentences
            sentences = [s.strip() for s in sentences if s.strip()]
            
            for i, sentence in enumerate(sentences):
                if not sentence:
                    continue
                
                # Word tokenization
                if lang == 'ko':
                    # Simple Korean word splitting (spaces and punctuation)
                    words = re.findall(r'[가-힣]+|[a-zA-Z]+|[0-9]+', sentence)
                else:
                    # Use NLTK for English
                    words = word_tokenize(sentence)
                
                # Filter out very short words
                words = [w for w in words if len(w) > 1]
                
                if words:  # Only include segments with words
                    segment = {
                        'sentence_id': i,
                        'text': sentence,
                        'words': words,
                        'word_count': len(words),
                        'char_count': len(sentence)
                    }
                    segments.append(segment)
            
        except Exception as e:
            self.logger.error(f"Error segmenting text: {e}")
            # Fallback: single segment
            segments = [{
                'sentence_id': 0,
                'text': text,
                'words': text.split(),
                'word_count': len(text.split()),
                'char_count': len(text)
            }]
        
        return segments



