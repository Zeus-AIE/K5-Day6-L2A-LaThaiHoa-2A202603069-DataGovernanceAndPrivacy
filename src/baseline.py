"""Syntactic/file-existence baseline for Track T06 (Dataset Semantic Integrity Gate).

Per docs/tracks/T06.md, the baseline only checks whether each candidate release's
``annotations.json`` can be parsed, has top-level COCO keys (``images``,
``annotations``, ``categories``), and points to image files that exist on disk.
It intentionally misses semantic corruptions such as bbox shift, class swap,
stale label maps, dropped frames, and ID collisions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.manifest import verify_manifest
from .common.review import (
    apply_overrides,
    load_review_decisions,
    make_review_queue,
    write_review_queue,
)
from .metrics.tracks import get_track


def _resolve_image_path(input_dir: Path, cand_dir: Path, file_name: str) -> Path:
    direct = input_dir / file_name
    if direct.exists():
        return direct
    return cand_dir / file_name


def load(ctx: RunContext) -> dict[str, Any]:
    """Verify manifest provenance and load the candidate release index."""
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(
            actor="baseline",
            action="provenance_problems",
            item_id=str(manifest),
            after=problems[:20],
        )
        if ctx.config.get("strict_provenance", True):
            raise ValueError(
                f"provenance check failed ({len(problems)} problems), first: {problems[:3]}"
            )
    if ctx.track != "t06":
        raise NotImplementedError(f"src.baseline currently implements track 't06', got {ctx.track!r}.")
    releases = pd.read_csv(ctx.input_dir / "releases.csv")
    return {"manifest": pd.read_csv(manifest) if manifest.exists() else None, "releases": releases}


def process(ctx: RunContext, data: dict[str, Any]) -> pd.DataFrame:
    """Run shallow syntactic + file-existence checks on each candidate release."""
    if ctx.track != "t06":
        raise NotImplementedError(f"src.baseline currently implements track 't06', got {ctx.track!r}.")
    rows: list[dict[str, Any]] = []
    for row in data["releases"].itertuples(index=False):
        cand_dir = ctx.input_dir / str(row.path)
        ann_path = cand_dir / "annotations.json"
        decision = "GO"
        root_cause = "clean"
        score = 0.0
        reason = "JSON valid, COCO top-level keys present, all referenced image files exist"
        evidence_path = ""
        try:
            with open(ann_path, "r", encoding="utf-8") as fh:
                coco = json.load(fh)
            if not isinstance(coco, dict) or not all(
                isinstance(coco.get(k), list) for k in ("images", "annotations", "categories")
            ):
                decision = "HOLD"
                root_cause = "corrupt_json"
                score = 1.0
                reason = "Missing or non-list COCO top-level keys (images/annotations/categories)"
            else:
                imgs = coco["images"]
                if imgs:
                    evidence_path = str(imgs[0].get("file_name", ""))
                missing = [
                    img.get("file_name", "")
                    for img in imgs
                    if not _resolve_image_path(ctx.input_dir, cand_dir, str(img.get("file_name", ""))).exists()
                ]
                if missing:
                    decision = "HOLD"
                    root_cause = "missing_file"
                    score = 1.0
                    reason = f"{len(missing)} referenced image file(s) missing on disk (e.g. {missing[0]})"
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            decision = "HOLD"
            root_cause = "corrupt_json"
            score = 1.0
            reason = f"Failed to read/parse annotations.json: {exc}"

        rows.append(
            {
                "release_id": str(row.release_id),
                "decision": decision,
                "root_cause": root_cause,
                "score": float(score),
                "split": str(getattr(row, "split", "eval")),
                "reason": reason,
                "suggested_action": "hold_release" if decision != "GO" else "approve_release",
                "evidence_path": evidence_path,
            }
        )
    return pd.DataFrame(rows)


def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Emit prioritized review queue ordered by anomaly score descending."""
    budget = int(ctx.param("budget", 10))
    q = make_review_queue(
        scored,
        id_col="release_id",
        score_col="score",
        budget=budget,
        reason_col="reason",
        action_col="suggested_action",
        slice_col="split",
        evidence_col="evidence_path",
    )
    write_review_queue(q, ctx.out("review_queue.csv"))
    return q


def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Apply optional human review overrides from review_queue_reviewed.csv."""
    reviewed_path = ctx.output_dir / "review_queue_reviewed.csv"
    if not reviewed_path.exists():
        return scored
    reviewed = load_review_decisions(reviewed_path)
    ctx.stats["n_reviewed"] = len(reviewed)
    for r in reviewed.itertuples(index=False):
        ctx.audit.log(
            actor=str(r.reviewer),
            action=f"review:{r.reviewer_decision}",
            item_id=str(r.item_id),
            reason=str(getattr(r, "review_note", "") or ""),
        )
    return apply_overrides(
        scored,
        reviewed,
        id_col="release_id",
        value_col="decision",
        audit=ctx.audit,
    )


def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """Return batch Release Gate Decision based on syntactic check results."""
    blocked = final[final["decision"].str.upper() != "GO"]
    owner = ctx.config.get("owner", "La Thai Hoa (Release Gate Owner)")
    if len(blocked) > 0:
        ids = ", ".join(blocked["release_id"].astype(str).tolist()[:10])
        return Decision(
            status="HOLD",
            reason_code="BASELINE_SYNTAX_DEFECTS",
            reason=f"Baseline blocked {len(blocked)}/{len(final)} release(s) due to syntax/file failures: {ids}",
            owner=owner,
            scope=ctx.track,
            evidence=[str(ctx.out("release_decisions.csv"))],
            metrics={"n_releases": len(final), "n_blocked": int(len(blocked))},
        )
    return Decision(
        status="GO",
        reason_code="BASELINE_SYNTAX_OK",
        reason=f"All {len(final)} releases passed syntactic JSON + file-existence checks",
        owner=owner,
        scope=ctx.track,
        evidence=[str(ctx.out("release_decisions.csv"))],
        metrics={"n_releases": len(final), "n_blocked": 0},
    )


def run(ctx: RunContext) -> None:
    """Run the 5-step T06 baseline pipeline."""
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    ordered_cols = spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]
    final[ordered_cols].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
