"""
Keywords, summary, entities, and frames extraction module.
"""

import json
import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

class Tagger:
    """Extract keywords, summary, entities, and frames from text."""
    
    def __init__(self):
        """Initialize the tagger."""
        self.logger = logging.getLogger(__name__)
        
        # Load spaCy models (fallback to basic if not available)
        try:
            self.nlp_ko = spacy.load("ko_core_news_sm")
        except OSError:
            self.nlp_ko = None
            self.logger.warning("Korean spaCy model not found. Using basic processing.")
        
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp_en = None
            self.logger.warning("English spaCy model not found. Using NLTK fallback.")
        
        # Frame patterns (simplified)
        self.frame_patterns = {
            'ko': {
                '강경 대응': ['강경', '단호', '경고', '위협', '보복', '제재'],
                '협상': ['협상', '대화', '회담', '합의', '타협'],
                '군사 행동': ['군사', '군대', '무기', '공격', '방어', '훈련'],
                '경제 제재': ['제재', '경제', '무역', '금융', '봉쇄'],
                '외교적 해결': ['외교', '정치', '해결', '평화', '협력']
            },
            'en': {
                'aggressive response': ['aggressive', 'firm', 'warning', 'threat', 'retaliation', 'sanctions'],
                'negotiation': ['negotiation', 'dialogue', 'talks', 'agreement', 'compromise'],
                'military action': ['military', 'army', 'weapon', 'attack', 'defense', 'training'],
                'economic sanctions': ['sanctions', 'economic', 'trade', 'financial', 'blockade'],
                'diplomatic solution': ['diplomatic', 'political', 'solution', 'peace', 'cooperation']
            }
        }
        
        # Actor patterns
        self.actor_patterns = {
            'ko': {
                '국가': ['국가', '정부', '당국', '국가A', '국가B'],
                '군사': ['군대', '군사', '국방부', '육군', '해군', '공군'],
                '기관': ['기관', '부처', '청', '위원회'],
                '조직': ['조직', '단체', '연합', '동맹']
            },
            'en': {
                'state': ['state', 'government', 'authority', 'country'],
                'military': ['military', 'army', 'defense', 'navy', 'air force'],
                'agency': ['agency', 'ministry', 'department', 'commission'],
                'organization': ['organization', 'group', 'alliance', 'coalition']
            }
        }
    
    def run(self, in_path: str, out_path: str) -> None:
        """
        Run the tagging process.
        
        Args:
            in_path: Input articles.norm.jsonl path
            out_path: Output kyw_sum.jsonl path
        """
        self.logger.info(f"Starting tagging: {in_path} -> {out_path}")
        
        processed_count = 0
        
        with open(in_path, 'r', encoding='utf-8') as infile, \
             open(out_path, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    article = json.loads(line.strip())
                    tagged = self._tag_article(article)
                    
                    if tagged:
                        outfile.write(json.dumps(tagged, ensure_ascii=False) + '\n')
                        processed_count += 1
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"Line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    self.logger.error(f"Line {line_num}: Processing error - {e}")
        
        self.logger.info(f"Tagging complete. Processed: {processed_count}")
    
    def _tag_article(self, article: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Tag a single article.
        
        Args:
            article: Normalized article dictionary
            
        Returns:
            Tagged article or None if failed
        """
        try:
            article_id = article.get('id', '')
            text = article.get('norm_text', '')
            lang = article.get('lang_final', 'en')
            segments = article.get('segments', [])
            
            if not text:
                self.logger.warning(f"Empty text for article {article_id}")
                return None
            
            # Extract keywords
            keywords = self._extract_keywords(text, lang)
            
            # Generate summary
            summary = self._generate_summary(text, lang, segments)
            
            # Extract entities and actors
            actors = self._extract_actors(text, lang)
            
            # Extract frames
            frames = self._extract_frames(text, lang)
            
            # Find span evidence
            span_evidence = self._find_span_evidence(text, keywords, frames, lang)
            
            # Build tagged record
            tagged = {
                'id': article_id,
                'kw': keywords,
                'summary': summary,
                'actors': actors,
                'frames': frames,
                'span_evidence': span_evidence
            }
            
            return tagged
            
        except Exception as e:
            self.logger.error(f"Error tagging article {article.get('id', 'unknown')}: {e}")
            return None
    
    def _extract_keywords(self, text: str, lang: str) -> List[str]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            List of keywords
        """
        try:
            # Use TF-IDF for keyword extraction
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words=self._get_stopwords(lang),
                ngram_range=(1, 2)
            )
            
            # Split text into sentences for TF-IDF
            if lang == 'ko':
                sentences = re.split(r'[.!?]\s*', text)
            else:
                sentences = sent_tokenize(text)
            
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return []
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top keywords
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-10:][::-1]
            
            keywords = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return keywords[:10]  # Limit to top 10
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _get_stopwords(self, lang: str) -> Set[str]:
        """Get stopwords for language."""
        try:
            if lang == 'ko':
                # Basic Korean stopwords
                return {'이', '그', '저', '것', '수', '등', '및', '또한', '그리고', '하지만'}
            else:
                return set(stopwords.words('english'))
        except:
            return set()
    
    def _generate_summary(self, text: str, lang: str, segments: List[Dict]) -> str:
        """
        Generate text summary.
        
        Args:
            text: Input text
            lang: Language code
            segments: Text segments
            
        Returns:
            Generated summary
        """
        try:
            # Simple extractive summarization
            if lang == 'ko':
                sentences = re.split(r'[.!?]\s*', text)
            else:
                sentences = sent_tokenize(text)
            
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 2:
                return text
            
            # Score sentences by word frequency
            word_freq = Counter()
            for sentence in sentences:
                words = self._tokenize_words(sentence, lang)
                word_freq.update(words)
            
            # Score sentences
            sentence_scores = []
            for sentence in sentences:
                words = self._tokenize_words(sentence, lang)
                score = sum(word_freq[word] for word in words)
                sentence_scores.append((sentence, score))
            
            # Select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            summary_sentences = [s[0] for s in sentence_scores[:2]]
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def _tokenize_words(self, text: str, lang: str) -> List[str]:
        """Tokenize text into words."""
        if lang == 'ko':
            # Simple Korean tokenization
            return re.findall(r'[가-힣]+|[a-zA-Z]+', text)
        else:
            return word_tokenize(text.lower())
    
    def _extract_actors(self, text: str, lang: str) -> List[str]:
        """
        Extract actors/entities from text.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            List of actors
        """
        actors = set()
        
        try:
            # Use spaCy if available
            nlp = self.nlp_ko if lang == 'ko' else self.nlp_en
            if nlp:
                doc = nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'NORP']:
                        actors.add(ent.text.strip())
            
            # Pattern-based extraction
            patterns = self.actor_patterns.get(lang, {})
            for category, keywords in patterns.items():
                for keyword in keywords:
                    if keyword in text:
                        # Extract surrounding context
                        pattern = rf'\b{re.escape(keyword)}\b'
                        matches = re.finditer(pattern, text, re.IGNORECASE)
                        for match in matches:
                            start = max(0, match.start() - 20)
                            end = min(len(text), match.end() + 20)
                            context = text[start:end].strip()
                            actors.add(context)
            
            return list(actors)[:10]  # Limit to top 10
            
        except Exception as e:
            self.logger.error(f"Error extracting actors: {e}")
            return []
    
    def _extract_frames(self, text: str, lang: str) -> List[str]:
        """
        Extract frames from text.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            List of frames with confidence
        """
        frames = []
        
        try:
            patterns = self.frame_patterns.get(lang, {})
            text_lower = text.lower()
            
            for frame_name, keywords in patterns.items():
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                if matches > 0:
                    confidence = min(1.0, matches / len(keywords))
                    frames.append({
                        'frame': frame_name,
                        'confidence': round(confidence, 2)
                    })
            
            # Sort by confidence
            frames.sort(key=lambda x: x['confidence'], reverse=True)
            return frames[:5]  # Top 5 frames
            
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
            return []
    
    def _find_span_evidence(self, text: str, keywords: List[str], frames: List[Dict], lang: str) -> List[List[int]]:
        """
        Find span evidence for keywords and frames.
        
        Args:
            text: Input text
            keywords: List of keywords
            frames: List of frames
            lang: Language code
            
        Returns:
            List of [start, end] positions
        """
        spans = []
        
        try:
            # Find keyword spans
            for keyword in keywords:
                pattern = re.escape(keyword)
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    spans.append([match.start(), match.end()])
            
            # Find frame spans
            for frame_info in frames:
                frame_name = frame_info['frame']
                patterns = self.frame_patterns.get(lang, {}).get(frame_name, [])
                for pattern in patterns:
                    matches = re.finditer(re.escape(pattern), text, re.IGNORECASE)
                    for match in matches:
                        spans.append([match.start(), match.end()])
            
            # Remove duplicates and sort
            spans = list(set(tuple(span) for span in spans))
            spans.sort(key=lambda x: x[0])
            
            return [list(span) for span in spans[:20]]  # Limit to 20 spans
            
        except Exception as e:
            self.logger.error(f"Error finding span evidence: {e}")
            return []



