"""
Content filtering and gating module.
"""

import json
import re
import logging
import yaml
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
from scipy import stats

class Gating:
    """Content filtering and gating based on indicator patterns."""
    
    def __init__(self):
        """Initialize the gating system."""
        self.logger = logging.getLogger(__name__)
        self.indicator_patterns = {}
        self.source_reputation = {}
        
    def run(self, in_path: str, out_path: str, weights_path: str = "config/weights.yaml") -> None:
        """
        Run the gating process.
        
        Args:
            in_path: Input kyw_sum.jsonl path
            out_path: Output gated.jsonl path
            weights_path: Path to weights configuration
        """
        self.logger.info(f"Starting gating: {in_path} -> {out_path}")
        
        # Load configuration
        self._load_weights(weights_path)
        
        # Collect statistics for z-score calculation
        all_scores = []
        processed_articles = []
        
        # First pass: collect data for statistics
        with open(in_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                try:
                    article = json.loads(line.strip())
                    score = self._calculate_indicator_score(article)
                    all_scores.append(score)
                    processed_articles.append((article, score))
                except Exception as e:
                    self.logger.error(f"Error processing article for statistics: {e}")
        
        if not all_scores:
            self.logger.warning("No articles processed for statistics")
            return
        
        # Calculate statistics
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        
        self.logger.info(f"Score statistics - Mean: {mean_score:.3f}, Std: {std_score:.3f}")
        
        # Second pass: apply gating
        processed_count = 0
        
        with open(out_path, 'w', encoding='utf-8') as outfile:
            for article, score in processed_articles:
                try:
                    gated = self._apply_gating(article, score, mean_score, std_score)
                    if gated:
                        outfile.write(json.dumps(gated, ensure_ascii=False) + '\n')
                        processed_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error applying gating: {e}")
        
        self.logger.info(f"Gating complete. Processed: {processed_count}")
    
    def _load_weights(self, weights_path: str) -> None:
        """Load indicator patterns and weights from configuration."""
        try:
            with open(weights_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.indicator_patterns = config.get('indicators', {})
            self.source_reputation = config.get('source_reputation', {})
            
            self.logger.info(f"Loaded {len(self.indicator_patterns)} indicator patterns")
            
        except FileNotFoundError:
            self.logger.warning(f"Weights file not found: {weights_path}. Using defaults.")
            self._load_default_patterns()
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            self._load_default_patterns()
    
    def _load_default_patterns(self) -> None:
        """Load default indicator patterns."""
        self.indicator_patterns = {
            '병력 이동': {
                'patterns': ['군사 이동', '병력 배치', '군대 이동', 'troop movement', 'military deployment'],
                'weight': 0.8,
                'confidence_threshold': 0.7
            },
            '무기 배치': {
                'patterns': ['무기 배치', '미사일 배치', 'weapon deployment', 'missile deployment'],
                'weight': 0.9,
                'confidence_threshold': 0.8
            },
            '경제 제재': {
                'patterns': ['제재', '경제 제재', 'sanctions', 'economic sanctions'],
                'weight': 0.7,
                'confidence_threshold': 0.6
            },
            '외교적 긴장': {
                'patterns': ['긴장', '대립', '갈등', 'tension', 'conflict', 'dispute'],
                'weight': 0.6,
                'confidence_threshold': 0.5
            },
            '군사 훈련': {
                'patterns': ['군사 훈련', '훈련', 'military exercise', 'training'],
                'weight': 0.5,
                'confidence_threshold': 0.4
            }
        }
        
        self.source_reputation = {
            'default': 0.8,
            'reuters': 0.95,
            'ap': 0.95,
            'bbc': 0.9,
            'cnn': 0.85,
            'fox': 0.8
        }
    
    def _calculate_indicator_score(self, article: Dict[str, Any]) -> float:
        """
        Calculate indicator score for an article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Indicator score
        """
        text = article.get('norm_text', '').lower()
        keywords = article.get('kw', [])
        frames = article.get('frames', [])
        
        total_score = 0.0
        hits = []
        
        # Check indicator patterns
        for indicator, config in self.indicator_patterns.items():
            patterns = config.get('patterns', [])
            weight = config.get('weight', 0.5)
            threshold = config.get('confidence_threshold', 0.5)
            
            # Pattern matching
            pattern_matches = 0
            for pattern in patterns:
                if pattern.lower() in text:
                    pattern_matches += 1
            
            # Keyword matching
            keyword_matches = 0
            for keyword in keywords:
                if any(pattern.lower() in keyword.lower() for pattern in patterns):
                    keyword_matches += 1
            
            # Frame matching
            frame_matches = 0
            for frame_info in frames:
                frame_name = frame_info.get('frame', '').lower()
                if any(pattern.lower() in frame_name for pattern in patterns):
                    frame_matches += 1
            
            # Calculate confidence
            total_matches = pattern_matches + keyword_matches + frame_matches
            if total_matches > 0:
                confidence = min(1.0, total_matches / len(patterns))
                
                if confidence >= threshold:
                    score = weight * confidence
                    total_score += score
                    
                    hits.append({
                        'indicator': indicator,
                        'val': 1,
                        'conf': round(confidence, 2),
                        'span': self._find_indicator_span(text, patterns)
                    })
        
        return total_score
    
    def _find_indicator_span(self, text: str, patterns: List[str]) -> List[int]:
        """Find span positions for indicator patterns."""
        for pattern in patterns:
            match = re.search(re.escape(pattern.lower()), text)
            if match:
                return [match.start(), match.end()]
        return [0, 0]
    
    def _apply_gating(self, article: Dict[str, Any], score: float, mean_score: float, std_score: float) -> Optional[Dict[str, Any]]:
        """
        Apply gating rules to an article.
        
        Args:
            article: Article dictionary
            score: Indicator score
            mean_score: Mean score for z-score calculation
            std_score: Standard deviation for z-score calculation
            
        Returns:
            Gated article or None if filtered out
        """
        try:
            # Calculate z-score
            if std_score > 0:
                z_score = (score - mean_score) / std_score
            else:
                z_score = 0.0
            
            # Calculate logistic score
            logit_score = 1 / (1 + np.exp(-z_score))
            
            # Apply source reputation adjustment
            source_rep = self._get_source_reputation(article)
            rep_adj = source_rep
            
            # Apply gating thresholds
            if score < 0.1:  # Very low indicator score
                return None
            
            if z_score < -2.0:  # Very low z-score
                return None
            
            # Build gated record
            gated = {
                'id': article.get('id', ''),
                'hits': self._extract_hits(article),
                'z': round(z_score, 2),
                'logit': round(logit_score, 2),
                'rep_adj': round(rep_adj, 2)
            }
            
            return gated
            
        except Exception as e:
            self.logger.error(f"Error applying gating: {e}")
            return None
    
    def _get_source_reputation(self, article: Dict[str, Any]) -> float:
        """Get source reputation score."""
        # This would typically use the URL or source information
        # For now, return default reputation
        return self.source_reputation.get('default', 0.8)
    
    def _extract_hits(self, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract indicator hits from article."""
        # This would extract the actual hits found during scoring
        # For now, return empty list
        return []
