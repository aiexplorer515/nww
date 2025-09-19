"""
NWW Main Runner - Complete Automation Package
(새 구조 + 호환 레이어 지원)
"""

import sys
import os
import argparse
import logging

# Add the package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------------
# Imports with compatibility layer
# -------------------------------------------------------------------
try:
    # New structure
    from nwwpkg.ingest.extract import Extractor
    from nwwpkg.preprocess.normalize import Normalizer
    from nwwpkg.analyze.tag import Tagger
    from nwwpkg.rules.gating import Gating

    from nwwpkg.score.score_is import ScoreIS
    from nwwpkg.score.score_dbn import ScoreDBN
    from nwwpkg.score.score_llm import LLMJudge

    from nwwpkg.fusion.calibrator import FusionCalibration
    from nwwpkg.eds.block_matching import BlockMatcher as EDSBlockMatcher
    from nwwpkg.scenario.scenario_builder import ScenarioBuilder

    from nwwpkg.ops.alert_decider import AlertDecider
    from nwwpkg.ops.ledger import Ledger as AuditLedger
    from nwwpkg.eventblock.aggregator import EventBlockAggregator

except ImportError as e:
    # Old structure fallback
    print(f"[WARN] Falling back to old imports: {e}")

    from nwwpkg.ingest import Extractor
    from nwwpkg.preprocess import Normalizer
    from nwwpkg.analyze import Tagger
    from nwwpkg.rules import Gating

    from nwwpkg.score import ScoreIS, ScoreDBN, LLMJudge
    from nwwpkg.fusion import FusionCalibration
    from nwwpkg.eds import EDSBlockMatcher
    from nwwpkg.scenario import ScenarioBuilder
    from nwwpkg.decider import AlertDecider
    from nwwpkg.ledger import AuditLedger
    from nwwpkg.eventblock import EventBlockAggregator


# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("nww.log", encoding="utf-8"),
        ],
    )


# -------------------------------------------------------------------
# Pipeline runner
# -------------------------------------------------------------------
def run_pipeline(bundle_dir: str, config_dir: str = "config") -> None:
    logger = logging.getLogger(__name__)
    logger.info(f"Starting NWW pipeline for bundle: {bundle_dir}")

    os.makedirs(bundle_dir, exist_ok=True)
    os.makedirs(os.path.join(bundle_dir, "logs"), exist_ok=True)

    try:
        # 1) Ingest
        logger.info("Step 1: Data Ingestion")
        extractor = Extractor(bundle_dir, os.path.join(config_dir, "sources.yaml"))
        extractor.run(os.path.join(bundle_dir, "articles.jsonl"))

        # 2) Normalize
        logger.info("Step 2: Text Normalization")
        normalizer = Normalizer()
        normalizer.run(
            os.path.join(bundle_dir, "articles.jsonl"),
            os.path.join(bundle_dir, "articles.norm.jsonl"),
            os.path.join(bundle_dir, "logs", "normalize.log"),
        )

        # 3) Analyze
        logger.info("Step 3: Text Analysis")
        tagger = Tagger()
        tagger.run(
            os.path.join(bundle_dir, "articles.norm.jsonl"),
            os.path.join(bundle_dir, "kyw_sum.jsonl"),
        )

        # 4) Gating
        logger.info("Step 4: Content Gating")
        gating = Gating()
        gating.run(
            os.path.join(bundle_dir, "kyw_sum.jsonl"),
            os.path.join(bundle_dir, "gated.jsonl"),
            os.path.join(config_dir, "weights.yaml"),
        )

        # 5) Scoring
        logger.info("Step 5: Multi-modal Scoring")

        logger.info("  - IS Scoring")
        score_is = ScoreIS()
        score_is.run(
            os.path.join(bundle_dir, "gated.jsonl"),
            os.path.join(bundle_dir, "scores.jsonl"),
            os.path.join(config_dir, "weights.yaml"),
        )

        logger.info("  - DBN Scoring")
        score_dbn = ScoreDBN()
        score_dbn.run(bundle_dir, os.path.join(bundle_dir, "scores.jsonl"))

        logger.info("  - LLM Scoring")
        score_llm = LLMJudge()
        score_llm.run(bundle_dir, os.path.join(bundle_dir, "scores.jsonl"))

        # 6) Fusion
        logger.info("Step 6: Score Fusion")
        fusion = FusionCalibration()
        fusion.run(
            os.path.join(bundle_dir, "scores.jsonl"),
            os.path.join(bundle_dir, "fused_scores.jsonl"),
        )

        # 7) EDS Block Matching
        logger.info("Step 7: EDS Block Matching")
        block_matcher = EDSBlockMatcher()
        block_matcher.run(
            bundle_dir,
            os.path.join(bundle_dir, "blocks.jsonl"),
        )

        # 8) Scenarios
        logger.info("Step 8: Scenario Construction")
        scenario_builder = ScenarioBuilder()
        scenario_builder.run(
            bundle_dir,
            os.path.join(bundle_dir, "scenarios.jsonl"),
        )

        # 9) Alerts
        logger.info("Step 9: Alert Generation")
        alert_decider = AlertDecider()
        alert_decider.run(
            bundle_dir,
            os.path.join(bundle_dir, "alerts.jsonl"),
        )

        # 10) Event Blocks
        logger.info("Step 10: Event Aggregation")
        event_aggregator = EventBlockAggregator()
        event_aggregator.run(
            bundle_dir,
            os.path.join(bundle_dir, "event_blocks.jsonl"),
        )

        # 11) Audit Ledger
        logger.info("Step 11: Audit Trail")
        ledger = AuditLedger()
        ledger.run(
            bundle_dir,
            os.path.join(bundle_dir, "ledger.jsonl"),
        )

        logger.info("✅ NWW pipeline completed successfully!")

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


# -------------------------------------------------------------------
# Main entry
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NWW - News World Watch")
    parser.add_argument(
        "--bundle", "-b", default="data/bundles/sample", help="Bundle directory path"
    )
    parser.add_argument(
        "--config", "-c", default="config", help="Configuration directory path"
    )
    parser.add_argument(
        "--log-level",
        "-l",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--ui", action="store_true", help="Launch Streamlit UI")

    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.ui:
        import subprocess

        ui_path = os.path.join(os.path.dirname(__file__), "ui", "app_total.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path])
    else:
        run_pipeline(args.bundle, args.config)


if __name__ == "__main__":
    main()
