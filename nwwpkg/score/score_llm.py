"""
LLM Judge scoring module.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any, Tuple
import openai
from datetime import datetime

class LLMJudge:
    """LLM-based scoring system."""
    
    def __init__(self, api_key_env: str = "OPENAI_API_KEY"):
        """
        Initialize the LLM Scorer.
        
        Args:
            api_key_env: Environment variable name for OpenAI API key
        """
        self.logger = logging.getLogger(__name__)
        
        # Setup OpenAI client
        api_key = os.getenv(api_key_env)
        if api_key:
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.logger.warning(f"OpenAI API key not found in {api_key_env}")
            self.client = None
        
        # Judge prompt template
        self.judge_prompt = """
You are an expert analyst evaluating news articles for crisis indicators. 
Analyze the provided evidence and determine the crisis level on a scale of 0.0 to 1.0.

Evidence:
{evidence}

Instructions:
1. Focus only on the provided evidence - do not introduce new facts
2. Evaluate the severity and credibility of crisis indicators
3. Consider the context and implications
4. Provide a score between 0.0 (no crisis) and 1.0 (severe crisis)
5. Explain your reasoning briefly

Score: [0.0-1.0]
Rationale: [Brief explanation]
"""
    
    def run(self, bundle_dir: str, scores_path: str, top_k: int = 5) -> None:
        """
        Run the LLM Scorer scoring process.
        
        Args:
            bundle_dir: Bundle directory path
            scores_path: Scores file path
            top_k: Number of top evidence sentences to use
        """
        self.logger.info(f"Starting LLM Judge scoring: {bundle_dir}")
        
        if not self.client:
            self.logger.error("OpenAI client not available. Skipping LLM Judge.")
            return
        
        # Load articles and analysis data
        articles = self._load_articles(bundle_dir)
        analysis_data = self._load_analysis_data(bundle_dir)
        
        processed_count = 0
        
        with open(scores_path, 'a', encoding='utf-8') as outfile:
            for article_id, article in articles.items():
                try:
                    # Get top evidence
                    evidence = self._extract_top_evidence(article, analysis_data.get(article_id, {}), top_k)
                    
                    if not evidence:
                        continue
                    
                    # Get LLM judgment
                    score, rationale, evidence_ids = self._get_llm_judgment(evidence)
                    
                    if score is not None:
                        # Build scored record
                        scored = {
                            'id': article_id,
                            'stage': 'LLM',
                            'score': round(score, 3),
                            'rationale': rationale,
                            'evidence_ids': evidence_ids,
                            'contra': score < 0.3  # Low score indicates contradictory evidence
                        }
                        
                        outfile.write(json.dumps(scored, ensure_ascii=False) + '\n')
                        processed_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing article {article_id}: {e}")
        
        self.logger.info(f"LLM Judge scoring complete. Processed: {processed_count}")
    
    def _load_articles(self, bundle_dir: str) -> Dict[str, Dict[str, Any]]:
        """Load normalized articles."""
        articles = {}
        articles_path = os.path.join(bundle_dir, "articles.norm.jsonl")
        
        try:
            with open(articles_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        article = json.loads(line.strip())
                        articles[article['id']] = article
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            self.logger.warning(f"Articles file not found: {articles_path}")
        
        return articles
    
    def _load_analysis_data(self, bundle_dir: str) -> Dict[str, Dict[str, Any]]:
        """Load analysis data (keywords, summary, etc.)."""
        analysis_data = {}
        analysis_path = os.path.join(bundle_dir, "kyw_sum.jsonl")
        
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        analysis_data[data['id']] = data
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            self.logger.warning(f"Analysis file not found: {analysis_path}")
        
        return analysis_data
    
    def _extract_top_evidence(self, article: Dict[str, Any], analysis: Dict[str, Any], top_k: int) -> List[Tuple[str, int]]:
        """
        Extract top evidence sentences.
        
        Args:
            article: Article data
            analysis: Analysis data
            top_k: Number of top sentences
            
        Returns:
            List of (sentence, sentence_id) tuples
        """
        text = article.get('norm_text', '')
        segments = article.get('segments', [])
        
        if not segments:
            # Fallback: simple sentence splitting
            sentences = text.split('. ')
            return [(sent, i) for i, sent in enumerate(sentences[:top_k]) if sent.strip()]
        
        # Score segments by relevance
        scored_segments = []
        keywords = analysis.get('kw', [])
        frames = analysis.get('frames', [])
        
        for segment in segments:
            sentence = segment.get('text', '')
            score = self._score_sentence_relevance(sentence, keywords, frames)
            scored_segments.append((sentence, segment.get('sentence_id', 0), score))
        
        # Sort by score and take top k
        scored_segments.sort(key=lambda x: x[2], reverse=True)
        return [(sent, sent_id) for sent, sent_id, _ in scored_segments[:top_k]]
    
    def _score_sentence_relevance(self, sentence: str, keywords: List[str], frames: List[Dict]) -> float:
        """Score sentence relevance based on keywords and frames."""
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Keyword scoring
        for keyword in keywords:
            if keyword.lower() in sentence_lower:
                score += 1.0
        
        # Frame scoring
        for frame_info in frames:
            frame_name = frame_info.get('frame', '').lower()
            if frame_name in sentence_lower:
                score += frame_info.get('confidence', 0.5)
        
        return score
    
    def _get_llm_judgment(self, evidence: List[Tuple[str, int]]) -> Tuple[Optional[float], str, List[int]]:
        """
        Get LLM judgment for evidence.
        
        Args:
            evidence: List of (sentence, sentence_id) tuples
            
        Returns:
            Tuple of (score, rationale, evidence_ids)
        """
        try:
            # Prepare evidence text
            evidence_text = "\n".join([f"{i+1}. {sent}" for i, (sent, _) in enumerate(evidence)])
            evidence_ids = [sent_id for _, sent_id in evidence]
            
            # Create prompt
            prompt = self.judge_prompt.format(evidence=evidence_text)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert crisis analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            score, rationale = self._parse_llm_response(response_text)
            
            return score, rationale, evidence_ids
            
        except Exception as e:
            self.logger.error(f"Error getting LLM judgment: {e}")
            return None, "Error in LLM processing", []
    
    def _parse_llm_response(self, response_text: str) -> Tuple[Optional[float], str]:
        """Parse LLM response to extract score and rationale."""
        try:
            lines = response_text.split('\n')
            score = None
            rationale = ""
            
            for line in lines:
                line = line.strip()
                if line.startswith('Score:'):
                    # Extract score
                    score_text = line.replace('Score:', '').strip()
                    # Remove brackets if present
                    score_text = score_text.replace('[', '').replace(']', '')
                    try:
                        score = float(score_text)
                        # Ensure score is in valid range
                        score = max(0.0, min(1.0, score))
                    except ValueError:
                        score = None
                elif line.startswith('Rationale:'):
                    rationale = line.replace('Rationale:', '').strip()
            
            # If no explicit rationale, use the rest of the response
            if not rationale and score is not None:
                rationale = response_text.replace(f"Score: {score}", "").strip()
            
            return score, rationale
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return None, "Error parsing response"
