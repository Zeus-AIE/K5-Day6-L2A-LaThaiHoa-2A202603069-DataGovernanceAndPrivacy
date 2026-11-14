"""Tests for Track T06 (Dataset Semantic Integrity & Provenance Gate) PoC & Baseline."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.common.review import load_review_decisions
from src.main import main
from src.metrics.tracks import eval_t06

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "fixtures" / "t06_release_gate"
CONFIG_PATH = ROOT / "configs" / "lab.yaml"

EXPECTED_DEFECTS = {
    "r01": ("stale_map", "dev"),
    "r02": ("dropped_frames", "dev"),
    "r05": ("bbox_shift", "dev"),
    "r07": ("class_swap", "dev"),
    "r08": ("id_collision", "dev"),
    "r09": ("stale_map", "dev"),
    "r10": ("bbox_shift", "dev"),
    "r16": ("dropped_frames", "eval"),
    "r18": ("id_collision", "eval"),
    "r20": ("class_swap", "eval"),
}
EXPECTED_CLEAN = {
    "r03": "dev",
    "r04": "dev",
    "r06": "dev",
    "r11": "eval",
    "r12": "eval",
    "r13": "eval",
    "r14": "eval",
    "r15": "eval",
    "r17": "eval",
    "r19": "eval",
}
_REVIEW_DTYPES = {
    "reviewer": str,
    "reviewer_decision": str,
    "override_value": str,
    "review_note": str,
    "reviewed_at": str,
}


def _write_self_gt(gt_path: Path) -> Path:
    rows = []
    for rid, (ctype, split) in sorted(EXPECTED_DEFECTS.items()):
        rows.append({"release_id": rid, "is_defective": 1, "corruption_type": ctype, "split": split})
    for rid, split in sorted(EXPECTED_CLEAN.items()):
        rows.append({"release_id": rid, "is_defective": 0, "corruption_type": "clean", "split": split})
    pd.DataFrame(rows).to_csv(gt_path, index=False)
    return gt_path


def _metric_map(eval_out: Any) -> dict[str, float | None]:
    return {m.name: (None if m.value is None else float(m.value)) for m in eval_out.metrics}


def test_t06_poc_end_to_end_and_deterministic(tmp_path: Path) -> None:
    out1 = tmp_path / "poc1"
    out2 = tmp_path / "poc2"
    rc1 = main(["--config", str(CONFIG_PATH), "--track", "t06", "--output", str(out1), "--run-name", "r1", "--offline", "--seed", "42"])
    rc2 = main(["--config", str(CONFIG_PATH), "--track", "t06", "--output", str(out2), "--run-name", "r2", "--offline", "--seed", "42"])
    assert rc1 == 0
    assert rc2 == 0

    for fname in (
        "release_decisions.csv",
        "review_queue.csv",
        "decision.json",
        "decision_log.csv",
        "audit_log.jsonl",
        "run_info.json",
        "contact_sheet.html",
        "yolo_roundtrip_report.json",
        "croissant_metadata.json",
        "validation_report.json",
    ):
        assert (out1 / fname).exists(), f"missing required output {fname}"

    assert (out1 / "release_decisions.csv").read_text(encoding="utf-8") == (
        out2 / "release_decisions.csv"
    ).read_text(encoding="utf-8")
    dec = json.loads((out1 / "decision.json").read_text(encoding="utf-8"))
    assert dec["status"] == "HOLD"
    assert "La Thai Hoa" in dec["owner"]


def test_t06_detects_all_five_corruptions_and_clean_releases(tmp_path: Path) -> None:
    out_dir = tmp_path / "poc"
    rc = main(["--config", str(CONFIG_PATH), "--track", "t06", "--output", str(out_dir), "--run-name", "poc", "--offline", "--seed", "42"])
    assert rc == 0

    pred_df = pd.read_csv(out_dir / "release_decisions.csv")
    by_id = pred_df.set_index("release_id")
    for rid, (ctype, _) in EXPECTED_DEFECTS.items():
        assert by_id.loc[rid, "decision"] != "GO"
        assert by_id.loc[rid, "root_cause"] == ctype
    for rid in EXPECTED_CLEAN:
        assert by_id.loc[rid, "decision"] == "GO"
        assert by_id.loc[rid, "root_cause"] == "clean"

    gt_path = _write_self_gt(tmp_path / "gt_releases.csv")
    res = _metric_map(eval_t06(gt_path, out_dir / "release_decisions.csv", {"split": "all"}))
    assert res["Silent-defect Detection F1"] == 1.0
    assert res["Defect recall"] == 1.0
    assert res["False-block rate"] == 0.0
    assert res["Root-cause accuracy"] == 1.0


def test_t06_baseline_misses_silent_corruptions(tmp_path: Path) -> None:
    out_dir = tmp_path / "baseline"
    rc = main(
        [
            "--track",
            "t06",
            "--pipeline",
            "src.baseline",
            "--output",
            str(out_dir),
            "--run-name",
            "baseline",
            "--offline",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    pred_df = pd.read_csv(out_dir / "release_decisions.csv")
    assert (pred_df["decision"] == "GO").all()

    gt_path = _write_self_gt(tmp_path / "gt_releases.csv")
    res = _metric_map(eval_t06(gt_path, out_dir / "release_decisions.csv", {"split": "all"}))
    assert res["Defect recall"] == 0.0
    assert res["Silent-defect Detection F1"] == 0.0


def test_t06_human_review_override_and_audit_trail(tmp_path: Path) -> None:
    out_dir = tmp_path / "poc"
    assert main(["--config", str(CONFIG_PATH), "--track", "t06", "--output", str(out_dir), "--run-name", "p1", "--offline", "--seed", "42"]) == 0

    q = pd.read_csv(out_dir / "review_queue.csv", dtype=_REVIEW_DTYPES)
    bad_q = q.copy()
    bad_q.loc[bad_q["item_id"] == "r01", "reviewer_decision"] = "edit"
    bad_path = tmp_path / "bad_review.csv"
    bad_q.to_csv(bad_path, index=False)
    with pytest.raises(ValueError):
        load_review_decisions(bad_path)

    q["reviewer"] = "La Thai Hoa"
    q.loc[q["item_id"] == "r01", ["reviewer_decision", "override_value", "review_note"]] = [
        "edit",
        "REWORK",
        "Route stale map to REWORK",
    ]
    q.loc[q["item_id"] == "r07", ["reviewer_decision", "override_value", "review_note"]] = [
        "approve",
        "",
        "Confirmed class swap",
    ]
    q.to_csv(out_dir / "review_queue_reviewed.csv", index=False)

    assert main(["--config", str(CONFIG_PATH), "--track", "t06", "--output", str(out_dir), "--run-name", "p2", "--offline", "--seed", "42"]) == 0
    pred = pd.read_csv(out_dir / "release_decisions.csv").set_index("release_id")
    assert pred.loc["r01", "decision"] == "REWORK"
    assert pred.loc["r07", "decision"] == "HOLD"

    audit_lines = [json.loads(line) for line in (out_dir / "audit_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    actions = {(entry.get("action"), entry.get("item_id")) for entry in audit_lines}
    assert ("override:edit", "r01") in actions
    assert ("review:approve", "r07") in actions


def test_t06_shock_test_corrupt_json_isolated_gracefully(tmp_path: Path) -> None:
    shock_input = tmp_path / "shock_input"
    shutil.copytree(FIXTURE_DIR, shock_input)
    r03_json = shock_input / "releases" / "candidates" / "r03" / "annotations.json"
    raw = r03_json.read_bytes()
    r03_json.write_bytes(raw[: len(raw) // 2])  # Instructor shock test: cut half the JSON

    # 1) Under strict provenance, post-manifest truncation halts cleanly with exit code 2 + audit entry
    strict_out = tmp_path / "shock_strict"
    rc_strict = main(
        [
            "--config",
            str(CONFIG_PATH),
            "--track",
            "t06",
            "--input",
            str(shock_input),
            "--output",
            str(strict_out),
            "--run-name",
            "shock_strict",
            "--offline",
            "--seed",
            "42",
        ]
    )
    assert rc_strict == 2
    assert (strict_out / "audit_log.jsonl").exists()
    assert "provenance_problems" in (strict_out / "audit_log.jsonl").read_text(encoding="utf-8")

    # 2) When scanning unsealed incoming candidates (strict_provenance: false), r03 is quarantined as corrupt_json
    nonstrict_cfg = tmp_path / "lab_nonstrict.yaml"
    track_abs = (ROOT / "configs" / "tracks" / "t06.yaml").as_posix()
    cfg_txt = CONFIG_PATH.read_text(encoding="utf-8")
    cfg_txt = cfg_txt.replace("tracks/t06.yaml", track_abs).replace("strict_provenance: true", "strict_provenance: false")
    nonstrict_cfg.write_text(cfg_txt, encoding="utf-8")
    shock_out = tmp_path / "shock_out"
    rc = main(
        [
            "--config",
            str(nonstrict_cfg),
            "--track",
            "t06",
            "--input",
            str(shock_input),
            "--output",
            str(shock_out),
            "--run-name",
            "shock",
            "--offline",
            "--seed",
            "42",
        ]
    )
    assert rc == 0
    pred = pd.read_csv(shock_out / "release_decisions.csv").set_index("release_id")
    assert pred.loc["r03", "decision"] == "HOLD"
    assert pred.loc["r03", "root_cause"] == "corrupt_json"
    assert pred.loc["r04", "decision"] == "GO"
    assert pred.loc["r07", "root_cause"] == "class_swap"


def test_t06_coco_yolo_roundtrip_and_croissant_export(tmp_path: Path) -> None:
    out_dir = tmp_path / "poc"
    assert main(["--config", str(CONFIG_PATH), "--track", "t06", "--output", str(out_dir), "--run-name", "poc", "--offline", "--seed", "42"]) == 0

    rt = json.loads((out_dir / "yolo_roundtrip_report.json").read_text(encoding="utf-8"))
    assert rt["passed"] is True
    assert rt["images_checked"] == 100
    assert rt["boxes_checked"] == 697
    assert rt["yolo_files_matched"] == 100
    assert rt["max_roundtrip_error_px"] <= 0.5

    croissant = json.loads((out_dir / "croissant_metadata.json").read_text(encoding="utf-8"))
    assert croissant["@type"] == "Dataset"
    assert croissant["conformsTo"] == "http://mlcommons.org/croissant/1.1"

    val = json.loads((out_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert val["total_releases"] == 20
    assert val["fully_passing_releases"] == 10
    assert val["blocked_releases"] == 10
