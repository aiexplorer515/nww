import json
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nwwpkg.utils.io import load_jsonl, save_jsonl


def cluster_events(events_df: pd.DataFrame, sim_thr: float = 0.35):
    """TF-IDF 기반으로 유사 이벤트 클러스터링"""
    texts = events_df["normalized"].fillna("").astype(str).tolist()
    if not any(texts):
        return events_df  # 빈 경우 그대로 반환

    vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
    X = vectorizer.fit_transform(texts)

    sims = cosine_similarity(X)
    cluster_ids = [-1] * len(events_df)
    cluster_id = 0

    for i in range(len(events_df)):
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = cluster_id
        for j in range(i + 1, len(events_df)):
            if sims[i, j] >= sim_thr:
                cluster_ids[j] = cluster_id
        cluster_id += 1

    events_df["cluster_id"] = cluster_ids
    return events_df


def build_blocks(events_df: pd.DataFrame):
    """클러스터 결과를 기반으로 blocks.jsonl 생성"""
    blocks = []
    for cid, group in events_df.groupby("cluster_id"):
        block = {
            "block_id": f"B{cid}",
            "block_label": group["block"].mode()[0] if "block" in group else "General",
            "num_events": len(group),
            "actors": list(group["actors"].explode().dropna().unique()) if "actors" in group else [],
            "frames": list(group["frames"].explode().dropna().unique()) if "frames" in group else [],
            "events": group.to_dict("records"),
        }
        blocks.append(block)
    return blocks


def accumulate_blocks(events_file: str, blocks_file: str, sim_thr: float = 0.35):
    """events.jsonl → blocks.jsonl 생성"""
    events_df = pd.DataFrame(load_jsonl(events_file))
    if events_df.empty:
        raise ValueError("❌ events.jsonl이 비어 있습니다.")

    events_df = cluster_events(events_df, sim_thr=sim_thr)
    blocks = build_blocks(events_df)

    save_jsonl(blocks_file, blocks)
    return events_df, pd.DataFrame(blocks)
