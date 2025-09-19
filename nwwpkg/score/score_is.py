"""
Indicator-based scoring module.
"""

import json
import logging
import yaml
from typing import Dict, List, Optional, Any
from collections import defaultdict

class ScoreIS:
    """Indicator-based scoring system."""
    
    def __init__(self):
        """Initialize the IS scorer."""
        self.logger = logging.getLogger(__name__)
        self.weights = {}
        
    def run(self, in_path: str, scores_path: str, weights_path: str = "config/weights.yaml") -> None:
        """
        Run the IS scoring process.
        
        Args:
            in_path: Input gated.jsonl path
            scores_path: Output scores.jsonl path
            weights_path: Path to weights configuration
        """
        self.logger.info(f"Starting IS scoring: {in_path} -> {scores_path}")
        
        # Load weights
        self._load_weights(weights_path)
        
        processed_count = 0
        
        with open(in_path, 'r', encoding='utf-8') as infile, \
             open(scores_path, 'a', encoding='utf-8') as outfile:
            
            for line in infile:
                try:
                    gated = json.loads(line.strip())
                    scored = self._score_article(gated)
                    
                    if scored:
                        outfile.write(json.dumps(scored, ensure_ascii=False) + '\n')
                        processed_count += 1
                        
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decode error: {e}")
                except Exception as e:
                    self.logger.error(f"Scoring error: {e}")
        
        self.logger.info(f"IS scoring complete. Processed: {processed_count}")
    
    def _load_weights(self, weights_path: str) -> None:
        """Load indicator weights from configuration."""
        try:
            with open(weights_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.weights = config.get('indicator_weights', {})
            
            if not self.weights:
                self._load_default_weights()
            
            self.logger.info(f"Loaded {len(self.weights)} indicator weights")
            
        except FileNotFoundError:
            self.logger.warning(f"Weights file not found: {weights_path}. Using defaults.")
            self._load_default_weights()
        except Exception as e:
            self.logger.error(f"Error loading weights: {e}")
            self._load_default_weights()
    
    def _load_default_weights(self) -> None:
        """Load default indicator weights."""
        self.weights = {
            '병력 이동': 0.8,
            '무기 배치': 0.9,
            '경제 제재': 0.7,
            '외교적 긴장': 0.6,
            '군사 훈련': 0.5,
            'troop movement': 0.8,
            'weapon deployment': 0.9,
            'economic sanctions': 0.7,
            'diplomatic tension': 0.6,
            'military exercise': 0.5
        }
    
    def _score_article(self, gated: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Score an article using indicator-based scoring.
        
        Args:
            gated: Gated article dictionary
            
        Returns:
            Scored article or None if failed
        """
        try:
            article_id = gated.get('id', '')
            hits = gated.get('hits', [])
            rep_adj = gated.get('rep_adj', 1.0)
            
            if not hits:
                return None
            
            # Calculate weighted score
            total_score = 0.0
            detail_scores = {}
            
            for hit in hits:
                indicator = hit.get('indicator', '')
                confidence = hit.get('conf', 0.0)
                value = hit.get('val', 1.0)
                
                # Get weight for indicator
                weight = self.weights.get(indicator, 0.5)
                
                # Calculate score: weight × hit × confidence × source_reputation
                score = weight * value * confidence * rep_adj
                total_score += score
                
                detail_scores[indicator] = round(score, 3)
            
            # Normalize score
            normalized_score = min(1.0, total_score)
            
            # Build scored record
            scored = {
                'id': article_id,
                'stage': 'IS',
                'score': round(normalized_score, 3),
                'detail': detail_scores,
                'rep_adj': round(rep_adj, 3)
            }
            
            return scored
            
        except Exception as e:
            self.logger.error(f"Error scoring article {gated.get('id', 'unknown')}: {e}")
            return None
