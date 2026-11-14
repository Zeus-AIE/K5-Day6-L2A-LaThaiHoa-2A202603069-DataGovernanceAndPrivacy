"""PoC Pipeline for Track T06: Dataset Semantic Integrity & Provenance Gate.

Implements the full 5-step release gate workflow (load -> process -> rank ->
review -> decide) against parent release ``releases/v1.0`` and label map
registry ``releases/label_maps.json``.

Also emits supporting governance & reproducibility artifacts per run:
- ``yolo_roundtrip_report.json`` (COCO <-> YOLO lossless conversion audit)
- ``croissant_metadata.json`` (Croissant 1.1 JSON-LD dataset metadata)
- ``validation_report.json`` (Great Expectations-style suite & per-release results)
- ``contact_sheet.html`` (visual evidence contact sheet with bbox overlays)
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .common.contact_sheet import render_contact_sheet
from .common.context import RunContext
from .common.decision import Decision, write_decision
from .common.io import read_json, sha256_file, write_json
from .common.manifest import verify_manifest
from .common.review import (
    apply_overrides,
    load_review_decisions,
    make_review_queue,
    write_review_queue,
)
from .metrics.tracks import get_track

_SEVERITY_PRIORITY = [
    "corrupt_json",
    "provenance_mismatch",
    "id_collision",
    "class_swap",
    "stale_map",
    "dropped_frames",
    "bbox_shift",
]


def _resolve_image_path(input_dir: Path, cand_dir: Path, file_name: str) -> Path:
    direct = input_dir / file_name
    if direct.exists():
        return direct
    return cand_dir / file_name


def _validate_coco_structure(coco: Any) -> str | None:
    """Validate minimal structural invariants of a parsed COCO dict."""
    if not isinstance(coco, dict):
        return "annotations.json root must be a JSON object"
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list):
            return f"missing or non-list COCO key '{key}'"
    for img in coco["images"]:
        if not isinstance(img, dict) or "id" not in img or "file_name" not in img:
            return "image record missing required id/file_name"
        if float(img.get("width", 0)) <= 0 or float(img.get("height", 0)) <= 0:
            return f"invalid image dimensions on image id={img.get('id')}"
    for ann in coco["annotations"]:
        if not isinstance(ann, dict) or not all(
            k in ann for k in ("id", "image_id", "category_id", "bbox")
        ):
            return "annotation record missing id/image_id/category_id/bbox"
        bbox = ann["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return f"invalid bbox format on annotation id={ann.get('id')}"
        for v in bbox:
            if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                return f"non-numeric/non-finite bbox coordinate on annotation id={ann.get('id')}"
    for cat in coco["categories"]:
        if not isinstance(cat, dict) or "id" not in cat or "name" not in cat:
            return "category record missing id/name"
    return None


def _build_parent_index(
    input_dir: Path, parent_rel: str
) -> dict[str, Any]:
    parent_dir = input_dir / parent_rel
    parent_ann_path = parent_dir / "annotations.json"
    parent_meta_path = parent_dir / "release_meta.json"
    parent_classes_path = parent_dir / "classes.txt"

    parent_coco = read_json(parent_ann_path)
    parent_meta = read_json(parent_meta_path) if parent_meta_path.exists() else {}
    classes = (
        [line.strip() for line in parent_classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if parent_classes_path.exists()
        else []
    )

    imgs_by_id = {int(img["id"]): img for img in parent_coco["images"]}
    anns_by_id = {int(ann["id"]): ann for ann in parent_coco["annotations"]}
    cats_by_id = {int(cat["id"]): str(cat["name"]) for cat in parent_coco["categories"]}

    return {
        "dir": parent_dir,
        "ann_path": parent_ann_path,
        "sha256": sha256_file(parent_ann_path),
        "coco": parent_coco,
        "meta": parent_meta,
        "classes": classes,
        "imgs_by_id": imgs_by_id,
        "anns_by_id": anns_by_id,
        "cats_by_id": cats_by_id,
        "yolo_dir": parent_dir / "yolo_labels",
    }


def _verify_coco_yolo_roundtrip(parent: dict[str, Any], tol_px: float) -> dict[str, Any]:
    """Convert parent COCO bboxes -> YOLO -> COCO and cross-check yolo_labels/*.txt."""
    coco = parent["coco"]
    imgs_by_id = parent["imgs_by_id"]
    yolo_dir: Path = parent["yolo_dir"]
    cats_sorted = sorted(parent["cats_by_id"].keys())
    cat_to_idx = {cid: idx for idx, cid in enumerate(cats_sorted)}

    anns_by_img: dict[int, list[dict[str, Any]]] = {img_id: [] for img_id in imgs_by_id}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    max_rt_err_px = 0.0
    max_file_diff = 0.0
    yolo_files_matched = 0
    boxes_checked = 0

    for img_id, img in imgs_by_id.items():
        w_img = float(img["width"])
        h_img = float(img["height"])
        encoded_lines: list[tuple[int, float, float, float, float]] = []

        for ann in anns_by_img.get(img_id, []):
            bx, by, bw, bh = [float(v) for v in ann["bbox"]]
            cls_idx = cat_to_idx[int(ann["category_id"])]
            xc = (bx + bw / 2.0) / w_img
            yc = (by + bh / 2.0) / h_img
            nw = bw / w_img
            nh = bh / h_img
            encoded_lines.append((cls_idx, xc, yc, nw, nh))

            # Round-trip decode back to COCO pixel coordinates
            dec_w = nw * w_img
            dec_h = nh * h_img
            dec_x = xc * w_img - dec_w / 2.0
            dec_y = yc * h_img - dec_h / 2.0
            err = max(abs(dec_x - bx), abs(dec_y - by), abs(dec_w - bw), abs(dec_h - bh))
            max_rt_err_px = max(max_rt_err_px, err)
            boxes_checked += 1

        stem = Path(str(img["file_name"])).stem
        yolo_file = yolo_dir / f"{stem}.txt"
        if yolo_file.exists():
            raw_lines = [ln.strip() for ln in yolo_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if len(raw_lines) == len(encoded_lines):
                file_ok = True
                for raw, (cls_idx, xc, yc, nw, nh) in zip(raw_lines, encoded_lines):
                    parts = raw.split()
                    if len(parts) != 5 or int(parts[0]) != cls_idx:
                        file_ok = False
                        break
                    diff = max(
                        abs(float(parts[1]) - xc),
                        abs(float(parts[2]) - yc),
                        abs(float(parts[3]) - nw),
                        abs(float(parts[4]) - nh),
                    )
                    max_file_diff = max(max_file_diff, diff)
                    if diff > 1e-4:
                        file_ok = False
                if file_ok:
                    yolo_files_matched += 1

    return {
        "parent_release": parent["meta"].get("release_id", "v1.0"),
        "images_checked": len(imgs_by_id),
        "boxes_checked": boxes_checked,
        "yolo_files_matched": yolo_files_matched,
        "max_roundtrip_error_px": round(max_rt_err_px, 8),
        "max_yolo_file_norm_diff": round(max_file_diff, 8),
        "roundtrip_tol_px": tol_px,
        "passed": bool(max_rt_err_px <= tol_px and yolo_files_matched == len(imgs_by_id)),
    }


def _select_boxes_for_image(
    coco: dict[str, Any],
    cand_cats: dict[int, str],
    target_img_id: int,
    parent_anns_by_id: dict[int, dict[str, Any]],
    root_cause: str,
) -> list[dict[str, Any]]:
    """Build bounding-box overlay descriptors for render_contact_sheet()."""
    boxes: list[dict[str, Any]] = []
    for ann in coco["annotations"]:
        if int(ann["image_id"]) != target_img_id:
            continue
        bx, by, bw, bh = [float(v) for v in ann["bbox"]]
        cat_name = cand_cats.get(int(ann["category_id"]), f"cat:{ann['category_id']}")
        color = "#d93025" if root_cause in ("class_swap", "stale_map", "id_collision") else "#1a73e8"
        p_ann = parent_anns_by_id.get(int(ann["id"]))
        if root_cause == "bbox_shift" and p_ann is not None:
            px, py, pw, ph = [float(v) for v in p_ann["bbox"]]
            shift = max(abs(bx - px), abs(by - py), abs(bw - pw), abs(bh - ph))
            if shift > 2.0:
                boxes.append(
                    {"x": px, "y": py, "w": pw, "h": ph, "label": f"v1.0:{cat_name}", "color": "#188038"}
                )
                color = "#d93025"
        boxes.append({"x": bx, "y": by, "w": bw, "h": bh, "label": cat_name, "color": color})
    return boxes[:12]


def _inspect_candidate_release(
    ctx: RunContext,
    row: Any,
    parent: dict[str, Any],
    label_maps: dict[str, dict[int, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Run all semantic, schema, geometry, and provenance gates on one candidate release."""
    release_id = str(row.release_id)
    rel_path = str(row.path)
    split = str(getattr(row, "split", "eval"))
    cand_dir = ctx.input_dir / rel_path
    ann_path = cand_dir / "annotations.json"
    meta_path = cand_dir / "release_meta.json"

    max_jitter_px = float(ctx.param("max_benign_jitter_px", 2.0))
    max_mean_shift_px = float(ctx.param("max_mean_shift_px", 1.5))

    expectations: dict[str, bool] = {
        "json_readable_and_valid_schema": False,
        "parent_sha256_matches": False,
        "unique_image_and_annotation_ids": False,
        "category_map_matches_version": False,
        "no_dropped_parent_frames": False,
        "bbox_drift_within_tolerance": False,
    }
    findings: list[tuple[str, float, str]] = []

    fallback_ev = ""
    if parent["coco"]["images"]:
        fallback_ev = str(parent["coco"]["images"][0]["file_name"])

    try:
        cand_coco = read_json(ann_path)
        cand_meta = read_json(meta_path) if meta_path.exists() else {}
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        findings.append(("corrupt_json", 1.0, f"Malformed or unreadable release JSON: {exc}"))
        rec = {
            "release_id": release_id,
            "decision": "HOLD",
            "root_cause": "corrupt_json",
            "score": 1.0,
            "split": split,
            "reason": findings[0][2],
            "suggested_action": "hold_and_quarantine",
            "evidence_path": fallback_ev,
            "n_images": 0,
            "n_annotations": 0,
            "max_bbox_shift_px": 0.0,
            "mean_bbox_shift_px": 0.0,
            "dup_ann_ids": 0,
            "missing_parent_frames": 0,
        }
        return rec, [], {"release_id": release_id, "expectations": expectations, "root_cause": "corrupt_json"}

    struct_err = _validate_coco_structure(cand_coco)
    if struct_err is not None:
        findings.append(("corrupt_json", 1.0, f"Schema integrity failure: {struct_err}"))
        rec = {
            "release_id": release_id,
            "decision": "HOLD",
            "root_cause": "corrupt_json",
            "score": 1.0,
            "split": split,
            "reason": findings[0][2],
            "suggested_action": "hold_and_quarantine",
            "evidence_path": fallback_ev,
            "n_images": len(cand_coco.get("images", [])) if isinstance(cand_coco, dict) else 0,
            "n_annotations": len(cand_coco.get("annotations", [])) if isinstance(cand_coco, dict) else 0,
            "max_bbox_shift_px": 0.0,
            "mean_bbox_shift_px": 0.0,
            "dup_ann_ids": 0,
            "missing_parent_frames": 0,
        }
        return rec, [], {"release_id": release_id, "expectations": expectations, "root_cause": "corrupt_json"}

    # Verify image files referenced by candidate exist on disk
    missing_files = [
        img["file_name"]
        for img in cand_coco["images"]
        if not _resolve_image_path(ctx.input_dir, cand_dir, str(img["file_name"])).exists()
    ]
    if missing_files:
        findings.append(("dropped_frames", 0.96, f"{len(missing_files)} image file(s) missing on disk"))

    expectations["json_readable_and_valid_schema"] = True

    # 1. Provenance check: parent SHA-256
    declared_sha = str(cand_meta.get("parent_sha256", ""))
    if declared_sha != parent["sha256"]:
        findings.append(
            (
                "provenance_mismatch",
                0.99,
                f"parent_sha256 mismatch: declared={declared_sha[:12]}... actual={parent['sha256'][:12]}...",
            )
        )
    else:
        expectations["parent_sha256_matches"] = True

    # 2. Unique ID check (MUST run before per-ann_id bbox/category comparison)
    img_ids = [int(img["id"]) for img in cand_coco["images"]]
    ann_ids = [int(ann["id"]) for ann in cand_coco["annotations"]]
    dup_img_ids = len(img_ids) - len(set(img_ids))
    dup_ann_ids = len(ann_ids) - len(set(ann_ids))
    if dup_img_ids > 0 or dup_ann_ids > 0:
        findings.append(
            (
                "id_collision",
                min(0.99, 0.94 + 0.002 * (dup_ann_ids + dup_img_ids)),
                f"Duplicate IDs detected: dup_ann_ids={dup_ann_ids}, dup_img_ids={dup_img_ids}",
            )
        )
    else:
        expectations["unique_image_and_annotation_ids"] = True

    # 3. Category ontology & label-map version integrity
    cand_cats = {int(c["id"]): str(c["name"]) for c in cand_coco["categories"]}
    lm_version = str(cand_meta.get("label_map_version", ""))
    expected_map = label_maps.get(lm_version)
    if expected_map is None:
        findings.append(("stale_map", 0.92, f"Unknown label_map_version '{lm_version}'"))
    elif cand_cats != expected_map:
        if set(cand_cats.values()) == set(expected_map.values()) and len(cand_cats) == len(expected_map):
            findings.append(
                (
                    "class_swap",
                    0.98,
                    f"Category IDs permuted vs {lm_version}: {cand_cats} != {expected_map}",
                )
            )
        else:
            findings.append(
                (
                    "stale_map",
                    0.92,
                    f"Categories {cand_cats} do not match declared {lm_version} ({expected_map})",
                )
            )
    else:
        expectations["category_map_matches_version"] = True

    # 4. Frame completeness vs parent & sequence contiguity
    cand_img_id_set = set(img_ids)
    missing_parent_imgs = sorted(set(parent["imgs_by_id"].keys()) - cand_img_id_set)
    orphan_anns = [ann for ann in cand_coco["annotations"] if int(ann["image_id"]) not in cand_img_id_set]
    seq_gaps: list[str] = []
    by_seq: dict[str, list[int]] = {}
    for img in cand_coco["images"]:
        if "sequence_id" in img and "frame_index" in img:
            by_seq.setdefault(str(img["sequence_id"]), []).append(int(img["frame_index"]))
    for seq_id, idxs in by_seq.items():
        if idxs:
            missing_idx = set(range(min(idxs), max(idxs) + 1)) - set(idxs)
            if missing_idx:
                seq_gaps.append(f"{seq_id}: missing frames {sorted(missing_idx)[:5]}")

    if missing_parent_imgs or orphan_anns or seq_gaps:
        findings.append(
            (
                "dropped_frames",
                min(0.97, 0.90 + 0.005 * len(missing_parent_imgs)),
                f"Dropped frames vs v1.0: missing_parent_imgs={len(missing_parent_imgs)}, "
                f"orphan_anns={len(orphan_anns)}, seq_gaps={seq_gaps[:2]}",
            )
        )
    else:
        expectations["no_dropped_parent_frames"] = True

    # 5. Bounding-box bounds & coordinate drift vs parent (only exact if IDs are unique)
    cand_imgs_by_id = {int(img["id"]): img for img in cand_coco["images"]}
    oob_boxes = 0
    for ann in cand_coco["annotations"]:
        bx, by, bw, bh = [float(v) for v in ann["bbox"]]
        img = cand_imgs_by_id.get(int(ann["image_id"]))
        if img is None:
            continue
        iw, ih = float(img["width"]), float(img["height"])
        if bw <= 0 or bh <= 0 or bx < -1e-6 or by < -1e-6 or (bx + bw) > iw + 1e-6 or (by + bh) > ih + 1e-6:
            oob_boxes += 1

    max_shift = 0.0
    mean_shift = 0.0
    max_shift_img_id: int | None = cand_coco["images"][0]["id"] if cand_coco["images"] else None
    cat_name_mismatches = 0

    if dup_ann_ids == 0 and dup_img_ids == 0:
        cand_anns_by_id = {int(ann["id"]): ann for ann in cand_coco["annotations"]}
        common_ids = set(cand_anns_by_id.keys()) & set(parent["anns_by_id"].keys())
        shifts: list[float] = []
        for aid in common_ids:
            ca = cand_anns_by_id[aid]
            pa = parent["anns_by_id"][aid]
            cx, cy, cw, ch = [float(v) for v in ca["bbox"]]
            px, py, pw, ph = [float(v) for v in pa["bbox"]]
            d = max(abs(cx - px), abs(cy - py), abs(cw - pw), abs(ch - ph))
            shifts.append(d)
            if d > max_shift:
                max_shift = d
                max_shift_img_id = int(ca["image_id"])
            c_name = cand_cats.get(int(ca["category_id"]), "")
            p_name = parent["cats_by_id"].get(int(pa["category_id"]), "")
            if c_name and p_name and c_name != p_name:
                cat_name_mismatches += 1
        if shifts:
            mean_shift = sum(shifts) / len(shifts)

    if oob_boxes > 0 or max_shift > max_jitter_px or mean_shift > max_mean_shift_px:
        findings.append(
            (
                "bbox_shift",
                min(0.97, 0.86 + max_shift / 400.0),
                f"BBox drift exceeds tolerance: max_shift={max_shift:.2f}px (limit {max_jitter_px}), "
                f"mean_shift={mean_shift:.2f}px (limit {max_mean_shift_px}), oob={oob_boxes}",
            )
        )
    else:
        expectations["bbox_drift_within_tolerance"] = True

    if cat_name_mismatches > 0 and not any(f[0] in ("class_swap", "stale_map") for f in findings):
        findings.append(
            (
                "class_swap",
                0.95,
                f"Per-annotation semantic class mismatches vs v1.0: {cat_name_mismatches} boxes",
            )
        )
        expectations["category_map_matches_version"] = False

    if findings:
        priority_map = {name: idx for idx, name in enumerate(_SEVERITY_PRIORITY)}
        findings.sort(key=lambda item: priority_map.get(item[0], 99))
        root_cause, score, reason = findings[0]
        decision = "HOLD"
        suggested_action = "rework_label_map" if root_cause == "stale_map" else "hold_and_fix"
    else:
        root_cause = "clean"
        score = round(min(0.05, mean_shift * 0.01), 4)
        decision = "GO"
        reason = (
            f"All 6 integrity & provenance gates passed "
            f"(max_jitter={max_shift:.2f}px <= {max_jitter_px}px)"
        )
        suggested_action = "approve_release"

    ev_img_id = max_shift_img_id
    if root_cause == "dropped_frames" and missing_parent_imgs:
        # Show neighbor frame right before the first dropped frame if present
        first_miss = min(missing_parent_imgs)
        ev_img_id = max(0, first_miss - 1)
    if ev_img_id is None or ev_img_id not in cand_imgs_by_id:
        ev_img_id = int(cand_coco["images"][0]["id"]) if cand_coco["images"] else 0

    ev_img = cand_imgs_by_id.get(ev_img_id)
    evidence_path = str(ev_img["file_name"]) if ev_img else fallback_ev
    boxes = _select_boxes_for_image(
        cand_coco, cand_cats, ev_img_id, parent["anns_by_id"], root_cause
    )

    rec = {
        "release_id": release_id,
        "decision": decision,
        "root_cause": root_cause,
        "score": round(float(score), 4),
        "split": split,
        "reason": reason,
        "suggested_action": suggested_action,
        "evidence_path": evidence_path,
        "n_images": len(cand_coco["images"]),
        "n_annotations": len(cand_coco["annotations"]),
        "max_bbox_shift_px": round(max_shift, 4),
        "mean_bbox_shift_px": round(mean_shift, 4),
        "dup_ann_ids": dup_ann_ids,
        "missing_parent_frames": len(missing_parent_imgs),
    }
    val_entry = {"release_id": release_id, "expectations": expectations, "root_cause": root_cause}
    return rec, boxes, val_entry


def _write_croissant_metadata(
    ctx: RunContext,
    parent: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> None:
    """Emit offline Croissant 1.1 JSON-LD metadata summarizing lineage and dataset files."""
    manifest_df = pd.read_csv(ctx.input_dir / "manifest.csv")
    licenses = sorted(manifest_df["license"].dropna().astype(str).unique().tolist())
    n_clean = sum(1 for r in summary_rows if r["decision"] == "GO")
    n_blocked = len(summary_rows) - n_clean

    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "conformsTo": "http://purl.org/dc/terms/conformsTo",
        },
        "@type": "Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": "t06-road-scene-release-gate",
        "description": (
            f"COCO detection dataset governed by T06 Semantic Integrity & Provenance Gate. "
            f"Parent release {parent['meta'].get('release_id', 'v1.0')} "
            f"({len(parent['coco']['images'])} images, {len(parent['coco']['annotations'])} boxes); "
            f"candidate gate evaluated {len(summary_rows)} releases ({n_clean} GO, {n_blocked} blocked)."
        ),
        "license": licenses[0] if licenses else "CC0-1.0",
        "version": str(parent["meta"].get("release_id", "v1.0")),
        "cr:citeAs": f"AI20K Day06 Data Governance & Privacy Lab — {ctx.config.get('group_id', 'G06')}",
        "distribution": [
            {
                "@type": "cr:FileObject",
                "@id": "parent-annotations",
                "name": "releases/v1.0/annotations.json",
                "encodingFormat": "application/json",
                "sha256": parent["sha256"],
            },
            {
                "@type": "cr:FileObject",
                "@id": "label-maps",
                "name": "releases/label_maps.json",
                "encodingFormat": "application/json",
                "sha256": sha256_file(ctx.input_dir / "releases" / "label_maps.json"),
            },
            {
                "@type": "cr:FileObject",
                "@id": "release-manifest",
                "name": "manifest.csv",
                "encodingFormat": "text/csv",
                "sha256": sha256_file(ctx.input_dir / "manifest.csv"),
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "release-decisions",
                "description": "Release-level gate verdicts and root causes",
                "field": [
                    {"@type": "cr:Field", "name": "release_id", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "decision", "dataType": "sc:Text"},
                    {"@type": "cr:Field", "name": "root_cause", "dataType": "sc:Text"},
                ],
            }
        ],
    }
    write_json(ctx.out("croissant_metadata.json"), croissant)


# ---------------------------------------------------------------- 01 load
def load(ctx: RunContext) -> dict[str, Any]:
    """Verify input provenance via manifest.csv and load T06 release lineage artifacts."""
    manifest = ctx.input_dir / "manifest.csv"
    problems = verify_manifest(manifest) if manifest.exists() else [{"problem": "no_manifest"}]
    if problems:
        ctx.audit.log(
            actor="pipeline",
            action="provenance_problems",
            item_id=str(manifest),
            after=problems[:20],
        )
        if ctx.config.get("strict_provenance", True):
            raise ValueError(
                f"provenance check failed ({len(problems)} problems), first: {problems[:3]}"
            )

    data: dict[str, Any] = {
        "manifest": pd.read_csv(manifest) if manifest.exists() else None,
    }
    if ctx.track != "t06":
        return data

    parent_rel = str(ctx.param("parent_release", "releases/v1.0"))
    raw_maps = read_json(ctx.input_dir / "releases" / "label_maps.json")
    label_maps = {
        ver: {int(cid): str(cname) for cid, cname in cmap.items()}
        for ver, cmap in raw_maps.items()
    }
    data["releases"] = pd.read_csv(ctx.input_dir / "releases.csv")
    data["label_maps"] = label_maps
    data["parent"] = _build_parent_index(ctx.input_dir, parent_rel)
    return data


# ---------------------------------------------------------------- 02 process
def process(ctx: RunContext, data: dict[str, Any]) -> pd.DataFrame:
    """Validate every candidate release against v1.0 and emit semantic gate scores."""
    spec = get_track(ctx.track)
    if ctx.track != "t06":
        raise NotImplementedError(
            f"Implement process() for {spec.track} ({spec.title}). It must produce a DataFrame with at least "
            f"{spec.pred_columns} (saved as out/{spec.pred_file}). See docs/tracks/{spec.track.upper()}.md"
        )

    parent = data["parent"]
    label_maps = data["label_maps"]
    rows: list[dict[str, Any]] = []
    boxes_by_item: dict[str, list[dict[str, Any]]] = {}
    val_entries: list[dict[str, Any]] = []

    for row in data["releases"].itertuples(index=False):
        rec, boxes, val_entry = _inspect_candidate_release(ctx, row, parent, label_maps)
        rows.append(rec)
        boxes_by_item[rec["release_id"]] = boxes
        val_entries.append(val_entry)

    rt_tol = float(ctx.param("roundtrip_tol_px", 0.5))
    rt_report = _verify_coco_yolo_roundtrip(parent, tol_px=rt_tol)
    write_json(ctx.out("yolo_roundtrip_report.json"), rt_report)
    write_json(ctx.out("evidence_boxes.json"), boxes_by_item)
    _write_croissant_metadata(ctx, parent, rows)

    passed_releases = sum(1 for v in val_entries if all(v["expectations"].values()))
    write_json(
        ctx.out("validation_report.json"),
        {
            "suite_name": "t06_semantic_release_gate_v1",
            "parent_release": parent["meta"].get("release_id", "v1.0"),
            "parent_sha256": parent["sha256"],
            "roundtrip_passed": rt_report["passed"],
            "total_releases": len(val_entries),
            "fully_passing_releases": passed_releases,
            "blocked_releases": len(val_entries) - passed_releases,
            "results": val_entries,
        },
    )
    ctx.stats["roundtrip_passed"] = rt_report["passed"]
    ctx.audit.log(
        actor="pipeline",
        action="release_gate_evaluated",
        item_id=ctx.track,
        after={
            "n_releases": len(rows),
            "n_blocked": len(val_entries) - passed_releases,
            "roundtrip_passed": rt_report["passed"],
        },
    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 03 rank
def rank(ctx: RunContext, scored: pd.DataFrame) -> pd.DataFrame:
    """Select top-B suspect releases by anomaly score and render HTML contact sheet."""
    budget = int(ctx.param("budget", 10))
    if "score" not in scored.columns:
        raise ValueError("rank(): process() output needs a numeric `score` column")
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

    boxes_path = ctx.output_dir / "evidence_boxes.json"
    boxes_by_item = read_json(boxes_path) if boxes_path.exists() else {}
    render_contact_sheet(
        q,
        ctx.out("contact_sheet.html"),
        image_root=ctx.input_dir,
        boxes_by_item=boxes_by_item,
        title="T06 Release Gate (Cổng Kiểm Soát Toàn Vẹn Dữ Liệu) — Nguồn ảnh: fixtures/t06_release_gate/images/ (synthetic:tools/make_fixtures.py, seed 42)",
    )
    return q


# ---------------------------------------------------------------- 04 review
def review(ctx: RunContext, scored: pd.DataFrame, queue: pd.DataFrame) -> pd.DataFrame:
    """Ingest human review decisions and record full audit trail."""
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


# ---------------------------------------------------------------- 05 decide
def decide(ctx: RunContext, final: pd.DataFrame) -> Decision:
    """Produce structured Release Gate Decision (HOLD if any silent defect blocked)."""
    owner = ctx.config.get("owner", "La Thai Hoa (Release Gate Owner / Data Steward)")
    blocked = final[final["decision"].str.upper() != "GO"]
    n_clean = int(len(final) - len(blocked))
    if len(blocked) > 0:
        breakdown = blocked["root_cause"].value_counts().to_dict() if "root_cause" in blocked.columns else {}
        blocked_ids = ", ".join(blocked["release_id"].astype(str).tolist())
        return Decision(
            status="HOLD",
            reason_code="SILENT_DEFECTS_BLOCKED",
            reason=(
                f"Blocked {len(blocked)}/{len(final)} candidate releases failing semantic/provenance "
                f"integrity gates ({blocked_ids}); approved {n_clean} clean releases."
            ),
            owner=owner,
            scope=ctx.track,
            evidence=[
                str(ctx.out("release_decisions.csv")),
                str(ctx.out("validation_report.json")),
                str(ctx.out("yolo_roundtrip_report.json")),
                str(ctx.out("croissant_metadata.json")),
            ],
            metrics={
                "n_releases": int(len(final)),
                "n_blocked": int(len(blocked)),
                "n_clean": n_clean,
                "root_cause_counts": {str(k): int(v) for k, v in breakdown.items()},
                "roundtrip_passed": bool(ctx.stats.get("roundtrip_passed", True)),
            },
        )
    return Decision(
        status="GO",
        reason_code="ALL_RELEASES_VERIFIED",
        reason=f"All {len(final)} candidate releases passed semantic integrity & provenance checks.",
        owner=owner,
        scope=ctx.track,
        evidence=[
            str(ctx.out("release_decisions.csv")),
            str(ctx.out("validation_report.json")),
            str(ctx.out("yolo_roundtrip_report.json")),
            str(ctx.out("croissant_metadata.json")),
        ],
        metrics={
            "n_releases": int(len(final)),
            "n_blocked": 0,
            "n_clean": n_clean,
            "roundtrip_passed": bool(ctx.stats.get("roundtrip_passed", True)),
        },
    )


def run(ctx: RunContext) -> None:
    """Execute the 5-step T06 release gate pipeline."""
    spec = get_track(ctx.track)
    data = load(ctx)
    scored = process(ctx, data)
    ctx.stats["n_items"] = len(scored)
    queue = rank(ctx, scored)
    final = review(ctx, scored, queue)
    ordered_cols = spec.pred_columns + [c for c in final.columns if c not in spec.pred_columns]
    final[ordered_cols].to_csv(ctx.out(spec.pred_file), index=False)
    write_decision(decide(ctx, final), ctx.out("decision.json"), ctx.out("decision_log.csv"))
