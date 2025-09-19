"""
Score fusion and calibration module.
"""

import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

class FusionCalibration:
    """Score fusion and calibration system."""
    
    def __init__(self):
        """Initialize the fusion system."""
        self.logger = logging.getLogger(__name__)
        self.calibrator = None
        
    def run(self, scores_path: str, output_path: str) -> None:
        """
        Run fusion and calibration.
        
        Args:
            scores_path: Input scores file path
            output_path: Output fused scores path
        """
        self.logger.info(f"Starting fusion: {scores_path} -> {output_path}")
        
        # Load scores
        scores = self._load_scores(scores_path)
        
        # Group by article ID
        grouped_scores = self._group_scores(scores)
        
        # Apply fusion
        fused_scores = []
        for article_id, article_scores in grouped_scores.items():
            fused = self._fuse_scores(article_scores)
            if fused:
                fused_scores.append(fused)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            for score in fused_scores:
                f.write(json.dumps(score, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Fusion complete. Processed: {len(fused_scores)}")
    
    def _load_scores(self, scores_path: str) -> List[Dict[str, Any]]:
        """Load scores from file."""
        scores = []
        try:
            with open(scores_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        scores.append(json.loads(line.strip()))
                    except:
                        continue
        except FileNotFoundError:
            self.logger.warning(f"Scores file not found: {scores_path}")
        return scores
    
    def _group_scores(self, scores: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group scores by article ID."""
        grouped = {}
        for score in scores:
            article_id = score.get('id', '')
            if article_id not in grouped:
                grouped[article_id] = []
            grouped[article_id].append(score)
        return grouped
    
    def _fuse_scores(self, article_scores: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Fuse scores for a single article."""
        if not article_scores:
            return None
        
        # Extract scores by stage
        stage_scores = {}
        for score in article_scores:
            stage = score.get('stage', '')
            stage_scores[stage] = score.get('score', 0.0)
        
        # Weighted fusion
        weights = {'IS': 0.4, 'DBN': 0.3, 'LLM': 0.3}
        fused_score = sum(weights.get(stage, 0.0) * score for stage, score in stage_scores.items())
        
        return {
            'id': article_scores[0].get('id', ''),
            'stage': 'FUSION',
            'score': round(fused_score, 3),
            'stage_scores': stage_scores,
            'weights': weights
        }



