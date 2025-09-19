def run(calibrated, out_dir="data/bundles/sample"):
    """Dummy conformal (add CI bounds)"""
    for r in calibrated:
        r["ci_low"] = max(0, r["score"] - 0.1)
        r["ci_high"] = min(1, r["score"] + 0.1)
    return calibrated
