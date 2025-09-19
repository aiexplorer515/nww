# nwwpkg/ingest/__init__.py
# -*- coding: utf-8 -*-
from . import news_collector

# 함수형 API 직노출(편의)
collect = news_collector.collect
get_domain = news_collector.get_domain

# 클래스 노출(백워드 호환)
try:
    from .news_collector import NewsCollector
except Exception:
    NewsCollector = None

# ✅ 레거시 호환: 일부 코드에서 Extractor 클래스를 import하는 케이스 지원
if NewsCollector is not None:
    Extractor = NewsCollector
else:
    # 폴백 shim: 같은 인터페이스 제공
    class Extractor:
        @staticmethod
        def collect(url: str):
            return news_collector.collect(url)
        @staticmethod
        def get_domain(url: str):
            return news_collector.get_domain(url)

__all__ = [
    "news_collector",
    "collect", "get_domain",
    "NewsCollector", "Extractor",
]



