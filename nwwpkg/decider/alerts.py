"""
Alert generation module.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

class AlertDecider:
    """Alert generation system."""
    
    def __init__(self):
        """Initialize the alert decider."""
        self.logger = logging.getLogger(__name__)
        self.threshold = 0.7
        
    def run(self, bundle_dir: str, output_path: str) -> None:
        """
        Run alert generation.
        
        Args:
            bundle_dir: Bundle directory path
            output_path: Output alerts file path
        """
        self.logger.info(f"Starting alert generation: {bundle_dir}")
        
        # Load fused scores
        scores_path = os.path.join(bundle_dir, "fused_scores.jsonl")
        scores = self._load_scores(scores_path)
        
        # Generate alerts
        alerts = []
        for score in scores:
            if score.get('score', 0.0) >= self.threshold:
                alert = {
                    'id': f"alert_{score.get('id', '')}",
                    'article_id': score.get('id', ''),
                    'score': score.get('score', 0.0),
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'severity': self._determine_severity(score.get('score', 0.0)),
                    'status': 'active'
                }
                alerts.append(alert)
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            for alert in alerts:
                f.write(json.dumps(alert, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Alert generation complete. Generated: {len(alerts)}")
    
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
    
    def _determine_severity(self, score: float) -> str:
        """Determine alert severity based on score."""
        if score >= 0.9:
            return 'critical'
        elif score >= 0.8:
            return 'high'
        elif score >= 0.7:
            return 'medium'
        else:
            return 'low'
