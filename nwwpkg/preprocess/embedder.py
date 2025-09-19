# -*- coding: utf-8 -*-
"""
임베더(embedder):
- 1순위: sentence-transformers (멀티링구얼)
- 폴백: TF-IDF
반환: List[List[float]]
"""
from __future__ import annotations
from typing import List

_model = None
_vectorizer = None

def _load_st_model():
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        # 멀티(ko/en) 준수 모델
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except Exception:
        _model = None
    return _model

def _load_tfidf():
    global _vectorizer
    if _vectorizer is not None:
        return _vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    _vectorizer = TfidfVectorizer(max_features=512)
    return _vectorizer

def embed(sentences: List[str]) -> List[List[float]]:
    if not sentences:
        return []
    # sentence-transformers 우선
    model = _load_st_model()
    if model:
        try:
            vecs = model.encode(sentences, normalize_embeddings=True)
            return vecs.tolist()
        except Exception:
            pass
    # 폴백: TF-IDF
    vect = _load_tfidf()
    X = vect.fit_transform(sentences)
    return X.toarray().tolist()
