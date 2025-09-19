import os
import logging
from nwwpkg.ingest.extract import Extractor
from nwwpkg.preprocess.normalize import Normalizer
from nwwpkg.analyze.tag import Tagger

logger = logging.getLogger(__name__)

def run_pipeline(stage: str, bundle_dir: str) -> str:
    """
    Dispatcher: stage 값에 따라 해당 단계 실행
    반환: 생성된 JSONL 파일 경로
    """
    os.makedirs(bundle_dir, exist_ok=True)

    if stage == "ingest":
        logger.info("📥 Ingest 실행")
        extractor = Extractor(bundle_dir)
        out_file = os.path.join(bundle_dir, "articles.jsonl")
        extractor.run(out_file)
        return out_file

    elif stage == "preprocess":
        logger.info("🧹 Preprocess 실행")
        normalizer = Normalizer()
        in_file = os.path.join(bundle_dir, "articles.jsonl")
        out_file = os.path.join(bundle_dir, "articles.norm.jsonl")
        log_file = os.path.join(bundle_dir, "logs", "normalize.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        normalizer.run(in_file, out_file, log_file)
        return out_file

    elif stage == "analysis":
        logger.info("🔍 Analysis 실행")
        tagger = Tagger()
        in_file = os.path.join(bundle_dir, "articles.norm.jsonl")
        out_file = os.path.join(bundle_dir, "kyw_sum.jsonl")
        tagger.run(in_file, out_file)
        return out_file

    elif stage == "scoring":
        logger.info("⚖️ Scoring 실행 (stub)")
        out_file = os.path.join(bundle_dir, "scores.jsonl")
        # TODO: 실제 scoring 모듈 연결
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("")
        return out_file

    elif stage == "scenarios":
        logger.info("📑 Scenario 생성 실행 (stub)")
        out_file = os.path.join(bundle_dir, "scenarios.jsonl")
        # TODO: 실제 scenario 모듈 연결
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("")
        return out_file

    elif stage == "alerts":
        logger.info("🚨 Alerts 생성 실행 (stub)")
        out_file = os.path.join(bundle_dir, "alerts.jsonl")
        # TODO: 실제 alerts 모듈 연결
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("")
        return out_file

    elif stage == "eds":
        logger.info("📚 EDS 실행 (stub)")
        out_file = os.path.join(bundle_dir, "eds.jsonl")
        # TODO: 실제 EDS 모듈 연결
        with open(out_file, "w", encoding="utf-8") as f:
            f.write("")
        return out_file

    else:
        raise ValueError(f"알 수 없는 단계: {stage}")


def run_full_pipeline(bundle_id: str, base_dir: str = "data/bundles") -> str:
    """
    전체 파이프라인 실행 (Ingest → Preprocess → Analysis → Scoring → Scenarios → Alerts → EDS)
    """
    bundle_dir = os.path.join(base_dir, bundle_id)
    stages = ["ingest", "preprocess", "analysis", "scoring", "scenarios", "alerts", "eds"]

    last_out = None
    for stage in stages:
        try:
            logger.info(f"▶️ {stage} 실행")
            last_out = run_pipeline(stage, bundle_dir)
        except Exception as e:
            logger.error(f"❌ {stage} 실패: {e}")
            break
    return last_out
