"""
Dynamic Bayesian Network scoring module.
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np

class ScoreDBN:
    """Dynamic Bayesian Network scoring system."""
    
    def __init__(self):
        """Initialize the DBN scorer."""
        self.logger = logging.getLogger(__name__)
        self.history_window = 2  # t-2, t-1, t
        
    def run(self, bundle_dir: str, scores_path: str) -> None:
        """
        Run the DBN scoring process.
        
        Args:
            bundle_dir: Bundle directory path
            scores_path: Scores file path
        """
        self.logger.info(f"Starting DBN scoring: {bundle_dir}")
        
        # Load historical scores
        historical_scores = self._load_historical_scores(scores_path)
        
        # Group scores by entity/region
        grouped_scores = self._group_scores_by_context(historical_scores)
        
        # Apply DBN correction
        processed_count = 0
        
        with open(scores_path, 'a', encoding='utf-8') as outfile:
            for context, scores in grouped_scores.items():
                try:
                    dbn_scores = self._apply_dbn_correction(scores)
                    
                    for score_record in dbn_scores:
                        outfile.write(json.dumps(score_record, ensure_ascii=False) + '\n')
                        processed_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error applying DBN correction for {context}: {e}")
        
        self.logger.info(f"DBN scoring complete. Processed: {processed_count}")
    
    def _load_historical_scores(self, scores_path: str) -> List[Dict[str, Any]]:
        """Load historical scores from file."""
        scores = []
        
        try:
            with open(scores_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        score_record = json.loads(line.strip())
                        scores.append(score_record)
                    except json.JSONDecodeError:
                        continue
                        
        except FileNotFoundError:
            self.logger.warning(f"Scores file not found: {scores_path}")
        
        return scores
    
    def _group_scores_by_context(self, scores: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group scores by entity/region context."""
        grouped = defaultdict(list)
        
        for score in scores:
            # Extract context (entity/region) from article ID or other fields
            context = self._extract_context(score)
            grouped[context].append(score)
        
        return dict(grouped)
    
    def _extract_context(self, score: Dict[str, Any]) -> str:
        """Extract context (entity/region) from score record."""
        # This would typically extract from article metadata
        # For now, use a simple grouping by ID prefix
        article_id = score.get('id', '')
        return article_id[:4] if len(article_id) >= 4 else 'default'
    
    def _apply_dbn_correction(self, scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply DBN correction to scores.
        
        Args:
            scores: List of historical scores for a context
            
        Returns:
            List of DBN-corrected scores
        """
        if len(scores) < 2:
            return []
        
        # Sort by timestamp (if available) or by order
        sorted_scores = sorted(scores, key=lambda x: x.get('id', ''))
        
        dbn_scores = []
        
        for i, current_score in enumerate(sorted_scores):
            if i < self.history_window:
                # Not enough history, use original score
                dbn_score = current_score.copy()
                dbn_score['stage'] = 'DBN'
                dbn_score['score_DBN'] = current_score.get('score', 0.0)
                dbn_scores.append(dbn_score)
            else:
                # Apply DBN correction
                historical_scores = [s.get('score', 0.0) for s in sorted_scores[i-self.history_window:i]]
                current_score_value = current_score.get('score', 0.0)
                
                # Simple DBN: weighted average with trend
                trend = self._calculate_trend(historical_scores)
                corrected_score = self._apply_trend_correction(current_score_value, trend)
                
                dbn_score = current_score.copy()
                dbn_score['stage'] = 'DBN'
                dbn_score['score_DBN'] = round(corrected_score, 3)
                dbn_score['trend'] = round(trend, 3)
                dbn_scores.append(dbn_score)
        
        return dbn_scores
    
    def _calculate_trend(self, historical_scores: List[float]) -> float:
        """Calculate trend from historical scores."""
        if len(historical_scores) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(historical_scores))
        y = np.array(historical_scores)
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _apply_trend_correction(self, current_score: float, trend: float) -> float:
        """Apply trend correction to current score."""
        # Adjust score based on trend
        # Positive trend increases score, negative trend decreases it
        correction_factor = 1.0 + (trend * 0.1)  # 10% adjustment per trend unit
        
        corrected_score = current_score * correction_factor
        
        # Ensure score stays within bounds
        return max(0.0, min(1.0, corrected_score))
