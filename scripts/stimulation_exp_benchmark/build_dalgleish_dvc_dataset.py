#!/usr/bin/env python3
"""Build an analysis-ready Dalgleish et al. photostimulation dataset for DVC.

This script follows a discovery-first workflow:
1. inspect the raw dataset and write a manifest
2. infer conceptual file roles with explicit reasoning
3. build trial-aligned neural summaries
4. perform leakage-safe preprocessing into copula-ready pseudo-observations
5. write clean tabular outputs plus metadata
6. run a minimal evaluation harness when possible

The implementation prefers readable, conservative heuristics over aggressive
guessing. Every important assumption is written to metadata and logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import struct
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm, rankdata

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None


LOGGER = logging.getLogger("dalgleish_dvc_dataset")
WINDOWS = {
    "baseline": (-1.0, -0.1),
    "stim": (0.0, 1.0),
    "post": (1.0, 2.0),
}
TRACE_PRIORITY = ("spks", "dff", "dff_traces", "F", "Fneu")
ROLE_NAMES = (
    "calcium_traces",
    "roi_metadata",
    "target_metadata",
    "timing_sync",
    "behavior",
    "ambiguous",
)


@dataclass
class TrialInference:
    session_id: str
    paq_file: str
    stim_source_path: Optional[str]
    frame_source_path: Optional[str]
    method: str
    time_unit_assumption: str
    frame_rate_hz: Optional[float]
    daq_sample_rate_hz: Optional[float]
    program_mapping_method: str
    warnings: List[str]


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def flatten_list(items: Iterable[Iterable[Any]]) -> List[Any]:
    out: List[Any] = []
    for item in items:
        out.extend(item)
    return out


def detect_mat_backend(path: Path) -> Dict[str, Any]:
    header = path.read_bytes()[:256]
    header_text = header.decode("latin1", errors="ignore")
    backend = "unknown"
    version = None
    if "MATLAB 7.3" in header_text or "HDF5 schema" in header_text:
        backend = "mat_v7.3_hdf5"
        version = "7.3"
    elif "MATLAB 5.0" in header_text:
        backend = "mat_v5"
        version = "5"
    return {
        "backend": backend,
        "version": version,
        "header_text": header_text.strip(),
    }


def _read_mat_tag(buf: bytes, offset: int) -> Tuple[int, int, int, int]:
    data_type, n_bytes = struct.unpack_from("<II", buf, offset)
    if data_type >> 16:
        small_n_bytes = data_type >> 16
        small_type = data_type & 0xFFFF
        return small_type, small_n_bytes, offset + 4, offset + 8
    data_offset = offset + 8
    next_offset = data_offset + ((n_bytes + 7) // 8) * 8
    return data_type, n_bytes, data_offset, next_offset


def _parse_mat_v5_matrix_summary(mat_bytes: bytes) -> Dict[str, Any]:
    classes = {
        1: "cell",
        2: "struct",
        3: "object",
        4: "char",
        5: "sparse",
        6: "double",
        7: "single",
        8: "int8",
        9: "uint8",
        10: "int16",
        11: "uint16",
        12: "int32",
        13: "uint32",
        14: "int64",
        15: "uint64",
    }
    cursor = 0
    try:
        _dt_flags, n_flags, off_flags, next_flags = _read_mat_tag(mat_bytes, cursor)
        flags = mat_bytes[off_flags : off_flags + n_flags]
        mx_class = struct.unpack_from("<I", flags, 0)[0] & 0xFF
        cursor = next_flags

        _dt_dims, n_dims, off_dims, next_dims = _read_mat_tag(mat_bytes, cursor)
        dims = list(struct.unpack_from("<" + "i" * (n_dims // 4), mat_bytes, off_dims)) if n_dims else []
        cursor = next_dims

        _dt_name, n_name, off_name, next_name = _read_mat_tag(mat_bytes, cursor)
        name = mat_bytes[off_name : off_name + n_name].decode("utf-8", errors="ignore")
        cursor = next_name

        summary = {"name": name, "dims": dims, "class": classes.get(mx_class, str(mx_class))}
        if mx_class == 2:
            _dt_len, _n_len, off_len, next_len = _read_mat_tag(mat_bytes, cursor)
            field_name_length = struct.unpack_from("<I", mat_bytes, off_len)[0]
            cursor = next_len
            _dt_fields, n_fields, off_fields, _next_fields = _read_mat_tag(mat_bytes, cursor)
            raw = mat_bytes[off_fields : off_fields + n_fields]
            field_names = []
            for start in range(0, len(raw), field_name_length):
                field = raw[start : start + field_name_length].split(b"\x00", 1)[0]
                field_names.append(field.decode("utf-8", errors="ignore"))
            summary["fields"] = field_names
        return summary
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": f"mat_v5_parse_error: {exc}"}


def _inspect_mat_v5_top_level(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"method": "pure_python_v5_header_parse", "top_level": [], "warnings": []}
    buf = path.read_bytes()
    cursor = 128
    while cursor + 8 <= len(buf):
        data_type, n_bytes, data_offset, next_offset = _read_mat_tag(buf, cursor)
        if data_type == 0 or next_offset > len(buf) + 8:
            break
        if data_type == 15:
            try:
                inflated = zlib.decompress(buf[data_offset : data_offset + n_bytes])
                inflated_cursor = 0
                while inflated_cursor + 8 <= len(inflated):
                    sub_type, sub_n_bytes, sub_offset, sub_next = _read_mat_tag(inflated, inflated_cursor)
                    if sub_type != 14 or sub_next > len(inflated) + 8:
                        break
                    out["top_level"].append(
                        _parse_mat_v5_matrix_summary(inflated[sub_offset : sub_offset + sub_n_bytes])
                    )
                    inflated_cursor = sub_next
            except Exception as exc:
                out["warnings"].append(f"Could not inflate compressed matrix in {path.name}: {exc}")
        elif data_type == 14:
            out["top_level"].append(_parse_mat_v5_matrix_summary(buf[data_offset : data_offset + n_bytes]))
        cursor = next_offset
    return out


def _extract_ascii_tokens(path: Path, min_len: int = 4, limit: int = 200) -> List[str]:
    data = path.read_bytes()
    tokens = re.findall(rb"[ -~]{%d,}" % min_len, data)
    unique: List[str] = []
    seen = set()
    for token in tokens:
        text = token.decode("ascii", errors="ignore").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
        if len(unique) >= limit:
            break
    return unique


def _decode_hdf5_char(dataset: Any) -> str:
    data = np.asarray(dataset[()])
    if data.dtype.kind in {"S", "U"}:
        return "".join(np.ravel(data).astype(str).tolist())
    if data.dtype.kind in {"i", "u"}:
        return "".join(chr(int(x)) for x in np.ravel(data) if int(x) > 0)
    return str(data)


def _bytes_attr(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, np.bytes_):
        return bytes(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        item = np.ravel(value)[0]
        if isinstance(item, (bytes, np.bytes_)):
            return bytes(item)
    return str(value).encode("utf-8", errors="ignore")


def _decode_hdf5_ref_item(item: Any, handle: Any) -> Any:
    if not isinstance(item, h5py.Reference):
        return item
    target = handle[item]
    matlab_class = _bytes_attr(target.attrs.get("MATLAB_class", b""))
    if isinstance(target, h5py.Dataset):
        if matlab_class == b"char":
            return _decode_hdf5_char(target)
        arr = np.asarray(target[()])
        if arr.dtype == h5py.ref_dtype or arr.dtype.kind == "O":
            return [_decode_hdf5_ref_item(x, handle) for x in arr.reshape(-1)]
        return arr
    return {key: _read_hdf5_node(target[key], handle, depth=1, max_depth=8) for key in target.keys()}


def _read_hdf5_node(node: Any, handle: Any, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return {"type": "max_depth_exceeded"}

    matlab_class = None
    if hasattr(node, "attrs") and "MATLAB_class" in node.attrs:
        raw = node.attrs["MATLAB_class"]
        if isinstance(raw, bytes):
            matlab_class = raw.decode("utf-8", errors="ignore")
        elif isinstance(raw, np.ndarray) and raw.size == 1:
            matlab_class = str(np.ravel(raw)[0])
        else:
            matlab_class = str(raw)

    if h5py is not None and isinstance(node, h5py.Dataset):
        if matlab_class == "char":
            return _decode_hdf5_char(node)
        data = node[()]
        if isinstance(data, np.ndarray) and (data.dtype == h5py.ref_dtype or data.dtype.kind == "O"):
            return [_decode_hdf5_ref_item(ref, handle) for ref in np.ravel(data)]
        if np.issubdtype(np.asarray(data).dtype, np.number):
            return np.asarray(data)
        return {
            "dtype": str(np.asarray(data).dtype),
            "shape": list(np.asarray(data).shape),
            "matlab_class": matlab_class,
        }

    if h5py is not None and isinstance(node, h5py.Group):
        out: Dict[str, Any] = {}
        for key in node.keys():
            out[key] = _read_hdf5_node(node[key], handle, depth + 1, max_depth)
        if matlab_class is not None:
            out["_matlab_class"] = matlab_class
        return out

    return {"type": str(type(node)), "matlab_class": matlab_class}


def _inspect_mat_v73(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"method": "ascii_fallback", "top_level": [], "warnings": []}
    if h5py is None:
        tokens = _extract_ascii_tokens(path, min_len=4, limit=500)
        top_level = []
        for token in tokens:
            low = token.lower()
            if token.startswith("/") or low in {"frames", "stims", "face_cam", "points"}:
                top_level.append({"path_or_token": token})
        out["top_level"] = top_level[:50]
        out["warnings"].append(
            "MATLAB v7.3/HDF5 file detected but h5py is unavailable; inspection is name-based only."
        )
        return out

    try:
        with h5py.File(path, "r") as handle:
            top_level = {}
            for key in handle.keys():
                top_level[key] = _read_hdf5_node(handle[key], handle, depth=0, max_depth=4)
            out["method"] = "h5py_recursive"
            out["top_level"] = top_level
            return out
    except Exception as exc:
        out["warnings"].append(f"Could not open HDF5 MAT file {path.name}: {exc}")
        tokens = _extract_ascii_tokens(path, min_len=4, limit=200)
        out["top_level"] = [{"path_or_token": token} for token in tokens[:50]]
        return out


def inspect_mat_file(path: Path) -> Dict[str, Any]:
    backend_info = detect_mat_backend(path)
    out: Dict[str, Any] = {
        "path": str(path),
        "backend": backend_info["backend"],
        "version": backend_info["version"],
        "header": backend_info["header_text"],
        "top_level": [],
        "reasoning": [],
        "roles": [],
        "warnings": [],
        "readable": True,
    }
    if backend_info["backend"] == "mat_v5":
        v5 = _inspect_mat_v5_top_level(path)
        out.update(v5)
    elif backend_info["backend"] == "mat_v7.3_hdf5":
        v73 = _inspect_mat_v73(path)
        out.update(v73)
    else:
        out["warnings"].append("Unrecognized MATLAB backend; inspection is incomplete.")

    roles, reasoning = infer_roles_for_file(path, out)
    out["roles"] = roles
    out["reasoning"] = reasoning
    return out


def _summary_items_from_top_level(top_level: Any) -> List[str]:
    items: List[str] = []
    if isinstance(top_level, list):
        for item in top_level:
            if isinstance(item, dict):
                name = item.get("name") or item.get("path_or_token") or item.get("path")
                dims = item.get("dims")
                if dims is not None:
                    items.append(f"{name} shape={dims}")
                elif name is not None:
                    items.append(str(name))
    elif isinstance(top_level, dict):
        for key, value in top_level.items():
            if key.startswith("_"):
                continue
            if isinstance(value, np.ndarray):
                items.append(f"{key} shape={list(value.shape)}")
            elif isinstance(value, dict):
                items.append(key)
            else:
                items.append(f"{key} ({type(value).__name__})")
    return items


def infer_roles_for_file(path: Path, inspection: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    top_level = inspection.get("top_level")
    summary_items = " | ".join(_summary_items_from_top_level(top_level)).lower()
    path_tokens = str(path).lower()
    roles: List[str] = []
    reasoning: List[str] = []

    def add(role: str, why: str) -> None:
        if role not in roles:
            roles.append(role)
            reasoning.append(why)

    if " f " in f" {summary_items} " or "f shape=" in summary_items:
        if "spks shape=" in summary_items or "fneu shape=" in summary_items:
            add(
                "calcium_traces",
                "Contains Suite2p-style arrays such as F/Fneu/spks with large ROI x frame matrices.",
            )
    if "iscell shape=" in summary_items or "stat shape=" in summary_items:
        add(
            "roi_metadata",
            "Contains Suite2p ROI metadata such as iscell/stat, which define usable ROIs and spatial footprints.",
        )
    if "points" in summary_items and ("groupcentroidx" in summary_items or "offsetx" in summary_items):
        add(
            "target_metadata",
            "Contains a NAPARM points struct with X/Y/Z and group centroid fields, consistent with stimulation targets.",
        )
    if "vf" in summary_items and ("sample_rate_hz" in summary_items or "laser_trials" in summary_items):
        add(
            "behavior",
            "Contains a behavior/training variable file with sample_rate_hz, laser_trials, protocol, and stim fields.",
        )
    if inspection.get("backend") == "mat_v7.3_hdf5":
        if "frames" in summary_items or "/paframes" in summary_items:
            add(
                "timing_sync",
                "HDF5 MAT file exposes frames/paframes entries, suggesting imaging-to-DAQ timing alignment.",
            )
        if "stims" in summary_items:
            add(
                "timing_sync",
                "HDF5 MAT file exposes stims entries, suggesting stimulation event timing metadata.",
            )
        if "face_cam" in summary_items or "/paface_cam" in summary_items:
            add(
                "behavior",
                "HDF5 MAT file exposes face_cam metadata, consistent with behavior/video side channels.",
            )
    if "cellpose" in path_tokens or "centroids" in path_tokens:
        add(
            "roi_metadata",
            "CellPose centroid or segmentation export indicates ROI/cell spatial metadata rather than trial timing.",
        )
    if not roles:
        add("ambiguous", "No strong modality signal beyond filename/context; leaving file marked ambiguous.")
    return roles, reasoning


def build_manifest(data_root: str | Path) -> Dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"data_root does not exist: {root}")

    sessions = []
    for path in sorted([p for p in root.iterdir() if p.is_dir()]):
        if (path / "Fall.mat").exists() or any(path.glob("*_paqanalysis.mat")):
            sessions.append(path)
    manifest: Dict[str, Any] = {
        "data_root": str(root),
        "n_sessions": len(sessions),
        "sessions": [],
        "global_warnings": [],
    }

    for session_path in sessions:
        session_entry: Dict[str, Any] = {
            "session_id": session_path.name,
            "path": str(session_path),
            "files": [],
            "role_map": {role: [] for role in ROLE_NAMES},
            "condensed_tree": [],
            "warnings": [],
        }

        for child in sorted(session_path.iterdir()):
            if child.is_dir():
                count = sum(1 for _ in child.rglob("*") if _.is_file())
                session_entry["condensed_tree"].append({"type": "dir", "name": child.name, "file_count": count})
            else:
                session_entry["condensed_tree"].append({"type": "file", "name": child.name})

        all_files = sorted([path for path in session_path.rglob("*") if path.is_file()])
        for file_path in all_files:
            file_entry: Dict[str, Any] = {
                "path": str(file_path),
                "suffix": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
            }
            if file_path.suffix.lower() == ".mat":
                file_entry["inspection"] = inspect_mat_file(file_path)
                roles = file_entry["inspection"]["roles"]
                for role in roles:
                    session_entry["role_map"][role].append(
                        {
                            "path": str(file_path),
                            "reasoning": file_entry["inspection"]["reasoning"],
                        }
                    )
                if file_entry["inspection"].get("warnings"):
                    session_entry["warnings"].extend(file_entry["inspection"]["warnings"])
            else:
                lower_name = file_path.name.lower()
                roles: List[str] = []
                reasoning: List[str] = []
                if file_path.suffix.lower() == ".npy" and "centroid" in lower_name:
                    roles = ["roi_metadata"]
                    reasoning = ["NumPy centroid export indicates ROI spatial metadata."]
                elif file_path.suffix.lower() == ".npy" and "seg" in lower_name:
                    roles = ["roi_metadata"]
                    reasoning = ["NumPy segmentation export indicates ROI masks or labels."]
                elif file_path.suffix.lower() in {".tif", ".tiff", ".png"}:
                    roles = ["roi_metadata"]
                    reasoning = ["Image export likely supports ROI/registration inspection rather than timing."]
                else:
                    roles = ["ambiguous"]
                    reasoning = ["Non-MAT file with no strong modality signal."]
                file_entry["inspection"] = {"roles": roles, "reasoning": reasoning}
                for role in roles:
                    session_entry["role_map"][role].append({"path": str(file_path), "reasoning": reasoning})
            session_entry["files"].append(file_entry)

        manifest["sessions"].append(session_entry)
    return manifest


def print_manifest_summary(manifest: Dict[str, Any]) -> None:
    LOGGER.info("Discovered %d session directories under %s", manifest["n_sessions"], manifest["data_root"])
    for session in manifest["sessions"]:
        LOGGER.info("Session %s", session["session_id"])
        tree_bits = []
        for item in session["condensed_tree"]:
            if item["type"] == "file":
                tree_bits.append(item["name"])
            else:
                tree_bits.append(f"{item['name']}/ ({item['file_count']} files)")
        LOGGER.info("  Tree: %s", ", ".join(tree_bits))

        for role in ROLE_NAMES:
            entries = session["role_map"].get(role, [])
            if not entries:
                continue
            LOGGER.info("  %s:", role)
            for entry in entries[:6]:
                LOGGER.info("    %s", Path(entry["path"]).name)
                for reason in entry.get("reasoning", []):
                    LOGGER.info("      why: %s", reason)
            if len(entries) > 6:
                LOGGER.info("      ... %d more", len(entries) - 6)

        if session.get("warnings"):
            unique_warnings = list(dict.fromkeys(session["warnings"]))
            for warning in unique_warnings[:5]:
                LOGGER.warning("  Warning: %s", warning)


def _extract_numeric_suffix(path: Path, token: str) -> Optional[int]:
    pattern = rf"_{token}_(\d+)"
    match = re.search(pattern, path.name)
    if match:
        return int(match.group(1))
    match = re.search(r"_(\d+)(?=\.mat$)", path.name)
    return int(match.group(1)) if match else None


def _matobj_to_python(obj: Any) -> Any:
    if hasattr(obj, "_fieldnames"):
        return {field: _matobj_to_python(getattr(obj, field)) for field in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype == object:
            flat = [_matobj_to_python(item) for item in obj.ravel()]
            try:
                return np.array(flat, dtype=object).reshape(obj.shape).tolist()
            except Exception:
                return flat
        return obj
    if isinstance(obj, (np.void,)):
        return obj
    return obj


def _safe_loadmat(path: Path, variable_names: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    return loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=variable_names)


def _get_scalar_candidates(obj: Any, path: str = "") -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.extend(_get_scalar_candidates(value, f"{path}.{key}" if path else key))
    elif isinstance(obj, np.ndarray):
        if obj.size == 1 and np.issubdtype(obj.dtype, np.number):
            out.append((path, float(np.asarray(obj).reshape(-1)[0])))
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        out.append((path, float(obj)))
    return out


def _extract_frame_rate_hz(ops: Dict[str, Any], vf: Optional[Dict[str, Any]]) -> Tuple[Optional[float], List[str]]:
    warnings: List[str] = []
    candidates = _get_scalar_candidates(ops)
    preferred_tokens = ("fs", "framerate", "frame_rate", "scanvolumerate", "volrate", "imagingrate")
    for token in preferred_tokens:
        for key, value in candidates:
            if token in key.lower() and 0.1 <= value <= 200.0:
                return float(value), warnings
    if vf is not None:
        for key, value in _get_scalar_candidates(vf):
            if "sample_rate_hz" in key.lower() and 0.1 <= value <= 500000.0:
                warnings.append(
                    "Only DAQ sample_rate_hz was found in behavior metadata; it is not used as imaging frame rate."
                )
                break
    warnings.append("Could not find an imaging frame rate in ops metadata.")
    return None, warnings


def _extract_daq_sample_rate_hz(vf: Optional[Dict[str, Any]]) -> Optional[float]:
    if vf is None:
        return None
    for key, value in _get_scalar_candidates(vf):
        if "sample_rate_hz" in key.lower() and 0.1 <= value <= 500000.0:
            return float(value)
    return None


def _get_trace_matrix(fall_path: Path) -> Tuple[np.ndarray, str]:
    inspect = inspect_mat_file(fall_path)
    names = [item.get("name") for item in inspect.get("top_level", []) if isinstance(item, dict)]
    selected_key = None
    for key in TRACE_PRIORITY:
        if key in names:
            selected_key = key
            break
    if selected_key is None:
        raise ValueError(f"Could not find a usable trace matrix in {fall_path}")
    loaded = _safe_loadmat(fall_path, variable_names=[selected_key])
    trace = np.asarray(loaded[selected_key], dtype=np.float32)
    return trace, selected_key


def _load_suite2p_metadata(fall_path: Path) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    loaded = _safe_loadmat(fall_path, variable_names=["iscell", "stat", "ops"])
    iscell = np.asarray(loaded.get("iscell"), dtype=np.float64)
    stat_raw = loaded.get("stat")
    ops_raw = loaded.get("ops")
    stat_python = _matobj_to_python(stat_raw)
    ops_python = _matobj_to_python(ops_raw)

    stat_list: List[Dict[str, Any]] = []
    if isinstance(stat_python, list):
        for item in stat_python:
            stat_list.append(item if isinstance(item, dict) else {"value": item})
    elif isinstance(stat_python, dict):
        stat_list.append(stat_python)
    elif isinstance(stat_python, np.ndarray):
        for item in stat_python.ravel():
            converted = _matobj_to_python(item)
            stat_list.append(converted if isinstance(converted, dict) else {"value": converted})
    return iscell, stat_list, ops_python if isinstance(ops_python, dict) else {"ops": ops_python}


def _roi_centers_from_stat(stat_list: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, stat in enumerate(stat_list):
        x = stat.get("xpix")
        y = stat.get("ypix")
        if x is None or y is None:
            rows.append({"roi_index": idx, "x_center": np.nan, "y_center": np.nan})
            continue
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        rows.append(
            {
                "roi_index": idx,
                "x_center": float(np.nanmean(x_arr)) if x_arr.size else np.nan,
                "y_center": float(np.nanmean(y_arr)) if y_arr.size else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _load_points_struct(path: Path) -> Dict[str, Any]:
    loaded = _safe_loadmat(path, variable_names=["points"])
    points = _matobj_to_python(loaded.get("points"))
    return points if isinstance(points, dict) else {"points": points}


def _load_behavior_struct(path: Path) -> Dict[str, Any]:
    loaded = _safe_loadmat(path, variable_names=["vf"])
    vf = _matobj_to_python(loaded.get("vf"))
    return vf if isinstance(vf, dict) else {"vf": vf}


def _extract_point_coordinates(points: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    x_raw = np.asarray(points.get("X", []), dtype=np.float64).reshape(-1)
    y_raw = np.asarray(points.get("Y", []), dtype=np.float64).reshape(-1)
    gcx = np.asarray(points.get("GroupCentroidX", []), dtype=np.float64).reshape(-1)
    gcy = np.asarray(points.get("GroupCentroidY", []), dtype=np.float64).reshape(-1)
    groups = np.asarray(points.get("Group", []), dtype=np.float64).reshape(-1)
    selected = np.asarray(points.get("Selected", []))

    # In this dataset the raw point arrays are dense spiral samples (often fixed at 200),
    # while the number of unique stimulation groups/centroids varies and is a better proxy
    # for targeted-neuron dose.
    if gcx.size and gcy.size and gcx.size == gcy.size:
        coords = np.unique(np.column_stack([gcx, gcy]), axis=0)
        x = coords[:, 0]
        y = coords[:, 1]
    elif groups.size and x_raw.size == groups.size and y_raw.size == groups.size:
        keep = []
        seen = set()
        for xx, yy, gg in zip(x_raw, y_raw, groups):
            key = int(gg)
            if key in seen:
                continue
            seen.add(key)
            keep.append((xx, yy))
        coords = np.asarray(keep, dtype=np.float64)
        x = coords[:, 0] if coords.size else np.array([], dtype=np.float64)
        y = coords[:, 1] if coords.size else np.array([], dtype=np.float64)
    else:
        x = x_raw
        y = y_raw
    if selected.size == x.size and selected.size > 0:
        mask = selected.astype(bool).reshape(-1)
        if mask.any():
            x = x[mask]
            y = y[mask]
    meta = {
        "n_points_raw": int(x_raw.size),
        "n_points_selected": int(x.size),
        "n_unique_groups": int(np.unique(groups).size) if groups.size else None,
        "group_centroid_x": np.asarray(points.get("GroupCentroidX", [])).reshape(-1).tolist(),
        "group_centroid_y": np.asarray(points.get("GroupCentroidY", [])).reshape(-1).tolist(),
    }
    return x, y, meta


def _build_target_catalog(
    session_path: Path,
    roi_centers: pd.DataFrame,
    vf: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[int, List[int]], List[str]]:
    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    targeted_roi_by_program: Dict[int, List[int]] = {}
    points_files = sorted(session_path.glob("targets/*_Points.mat"))
    if roi_centers.empty:
        warnings.append("ROI center table is empty; direct target-to-ROI mapping is unavailable.")

    points_by_naparm: Dict[int, Dict[str, Any]] = {}
    for path in points_files:
        naparm_id = _extract_numeric_suffix(path, "NAPARM")
        if naparm_id is None:
            continue
        points = _load_points_struct(path)
        points_by_naparm[naparm_id] = {
            "points_file": path,
            "points": points,
        }

    def _map_targets_to_rois(x: np.ndarray, y: np.ndarray, label: str) -> List[int]:
        dose = int(x.size)
        roi_indices: List[int] = []
        if dose > 0 and not roi_centers.empty:
            valid = roi_centers.dropna(subset=["x_center", "y_center"]).copy()
            if not valid.empty:
                roi_xy = valid[["x_center", "y_center"]].to_numpy(dtype=np.float64)
                target_xy = np.column_stack([x, y]).astype(np.float64)
                distances = np.sqrt(((target_xy[:, None, :] - roi_xy[None, :, :]) ** 2).sum(axis=2))
                nearest = np.argmin(distances, axis=1)
                min_dist = distances[np.arange(distances.shape[0]), nearest]
                mapped = valid.iloc[nearest]["roi_index"].to_numpy(dtype=int)
                roi_indices = sorted({int(roi) for roi, dist in zip(mapped, min_dist) if np.isfinite(dist) and dist <= 15.0})
                if len(roi_indices) < dose:
                    warnings.append(f"{label}: only mapped {len(roi_indices)} of {dose} target coordinates within 15 px.")
        return roi_indices

    if vf is not None and isinstance(vf.get("phasemasks"), dict) and isinstance(vf.get("protocol"), dict):
        phasemasks = vf["phasemasks"]
        protocol = vf["protocol"]
        details = np.asarray(phasemasks.get("phasemasks_details", []), dtype=object)
        stim_var_mat = np.asarray(protocol.get("stim_var_mat", []), dtype=np.int64)
        protocol_table = np.asarray(protocol.get("protocol_table", []), dtype=object)

        target_rows_by_idx: Dict[int, Dict[str, Any]] = {}
        for idx in range(len(details)):
            row = np.asarray(details[idx]).reshape(-1)
            if row.size == 0:
                continue
            label = str(row[0])
            naparm_match = re.search(r"NAPARM_(\d{3})", label)
            group_match = re.match(r"(\d{3})_", label)
            if naparm_match is None or group_match is None:
                continue
            naparm_id = int(naparm_match.group(1))
            group_id = int(group_match.group(1))
            if naparm_id not in points_by_naparm:
                continue
            points = points_by_naparm[naparm_id]["points"]
            all_x = np.asarray(points.get("X", []), dtype=np.float64).reshape(-1)
            all_y = np.asarray(points.get("Y", []), dtype=np.float64).reshape(-1)
            all_z = np.asarray(points.get("Z", []), dtype=np.float64).reshape(-1)
            groups = np.asarray(points.get("Group", []), dtype=np.int64).reshape(-1)
            group_mask = groups == group_id
            xyz = np.column_stack([all_x[group_mask], all_y[group_mask], all_z[group_mask]]).astype(np.float64)
            target_rows_by_idx[idx + 1] = {
                "naparm_id": naparm_id,
                "group_id": group_id,
                "points_file": str(points_by_naparm[naparm_id]["points_file"]),
                "targets_xyz": xyz,
                "dose_hint": int(xyz.shape[0]),
                "label": label,
            }

        for row_idx in range(stim_var_mat.shape[0]):
            stim_type = int(stim_var_mat[row_idx, 0])
            stim_var_zero = int(stim_var_mat[row_idx, 1])
            stim_var = stim_var_zero + 1
            protocol_entry = protocol_table[row_idx, 2] if protocol_table.ndim >= 2 and protocol_table.shape[1] >= 3 else np.nan
            phasemask_ids: List[int] = []
            if isinstance(protocol_entry, str):
                phasemask_ids = [int(tok) for tok in protocol_entry.split() if tok.strip()]
            elif isinstance(protocol_entry, (np.floating, float)) and np.isnan(protocol_entry):
                phasemask_ids = []
            elif isinstance(protocol_entry, (np.integer, int)):
                phasemask_ids = [int(protocol_entry)]

            coords = []
            z_coords = []
            points_file_list = []
            labels = []
            for pm_id in phasemask_ids:
                row_info = target_rows_by_idx.get(pm_id)
                if row_info is None:
                    continue
                coords.append(row_info["targets_xyz"][:, :2])
                z_coords.append(row_info["targets_xyz"][:, 2:3])
                points_file_list.append(row_info["points_file"])
                labels.append(row_info["label"])

            if coords:
                xy = np.vstack(coords)
                z = np.vstack(z_coords)
                x = xy[:, 0]
                y = xy[:, 1]
                dose = int(x.shape[0])
            else:
                x = np.array([], dtype=np.float64)
                y = np.array([], dtype=np.float64)
                z = np.array([], dtype=np.float64)
                dose = 0

            program_id = stim_type * 100 + stim_var
            roi_indices = _map_targets_to_rois(x, y, f"stim_type={stim_type}, stim_var={stim_var}")
            targeted_roi_by_program[program_id] = roi_indices
            rows.append(
                {
                    "program_id": program_id,
                    "stim_type": stim_type,
                    "stim_var": stim_var,
                    "points_file": points_file_list,
                    "dose": dose,
                    "target_x": x.tolist(),
                    "target_y": y.tolist(),
                    "target_z": z.reshape(-1).tolist() if np.asarray(z).size else [],
                    "targeted_roi_indices": roi_indices,
                    "metadata": {"phasemask_ids": phasemask_ids, "labels": labels},
                }
            )
    else:
        for path in points_files:
            program_id = _extract_numeric_suffix(path, "NAPARM")
            points = _load_points_struct(path)
            x, y, meta = _extract_point_coordinates(points)
            dose = int(x.size)
            roi_indices = _map_targets_to_rois(x, y, path.name)
            if program_id is not None:
                targeted_roi_by_program[program_id] = roi_indices
            rows.append(
                {
                    "program_id": program_id,
                    "stim_type": np.nan,
                    "stim_var": np.nan,
                    "points_file": str(path),
                    "dose": dose,
                    "target_x": x.tolist(),
                    "target_y": y.tolist(),
                    "target_z": [],
                    "targeted_roi_indices": roi_indices,
                    "metadata": meta,
                }
            )
    catalog = pd.DataFrame(
        rows,
        columns=[
            "program_id",
            "points_file",
            "dose",
            "target_x",
            "target_y",
            "targeted_roi_indices",
            "metadata",
        ],
    )
    return catalog, targeted_roi_by_program, warnings


def _flatten_numeric_arrays(obj: Any, path: str = "") -> List[Tuple[str, np.ndarray]]:
    out: List[Tuple[str, np.ndarray]] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = f"{path}/{key}" if path else key
            out.extend(_flatten_numeric_arrays(value, next_path))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_path = f"{path}[{idx}]"
            out.extend(_flatten_numeric_arrays(value, next_path))
    elif isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.number):
            out.append((path, np.asarray(obj)))
        elif obj.dtype == object:
            for idx, value in enumerate(obj.ravel()):
                out.extend(_flatten_numeric_arrays(value, f"{path}[{idx}]"))
    return out


def _normalize_event_candidates(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim > 1:
        if 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.shape[1] == 2:
            arr = arr[:, 0]
        else:
            arr = arr.reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr.astype(np.float64)
    unique = np.unique(arr)
    if unique.size <= 3 and np.all(np.isin(unique, [0, 1])):
        arr = arr.astype(np.int64)
        rising = np.flatnonzero(np.diff(np.r_[0, arr]) > 0)
        return rising.astype(np.float64)
    return np.sort(np.asarray(arr, dtype=np.float64))


def _select_candidate_array(flat_arrays: List[Tuple[str, np.ndarray]], tokens: Sequence[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    best_path = None
    best_arr = None
    best_score = -1.0
    for path, arr in flat_arrays:
        low = path.lower()
        score = 0.0
        for token in tokens:
            if token in low:
                score += 5.0
        if arr.ndim == 1 or 1 in arr.shape:
            score += 2.0
        if arr.size >= 3:
            score += 1.0
        if np.issubdtype(arr.dtype, np.number):
            score += 1.0
        if score > best_score:
            best_score = score
            best_path = path
            best_arr = np.asarray(arr)
    return best_path, best_arr


def _extract_program_ids_from_paq(
    flat_arrays: List[Tuple[str, np.ndarray]],
    n_events: int,
    available_program_ids: Sequence[int],
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    allowed = set(int(x) for x in available_program_ids if x is not None)
    for path, arr in flat_arrays:
        arr = np.asarray(arr).reshape(-1)
        if arr.size != n_events or arr.size == 0:
            continue
        if not np.all(np.isfinite(arr)):
            continue
        rounded = np.round(arr).astype(int)
        uniques = set(np.unique(rounded).tolist())
        if uniques and uniques.issubset(allowed):
            return path, rounded
    return None, None


def _load_paqanalysis(path: Path) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    warnings: List[str] = []
    backend = detect_mat_backend(path)["backend"]
    if backend == "mat_v5":
        try:
            loaded = _safe_loadmat(path)
            py = {key: _matobj_to_python(value) for key, value in loaded.items() if not key.startswith("__")}
            return py, warnings
        except Exception as exc:
            warnings.append(f"Could not load v5 paqanalysis file {path.name}: {exc}")
            return None, warnings
    if backend == "mat_v7.3_hdf5":
        if h5py is None:
            warnings.append(
                f"{path.name} is MATLAB v7.3/HDF5 but h5py is unavailable; trial extraction from this file is skipped."
            )
            return None, warnings
        try:
            with h5py.File(path, "r") as handle:
                py = {key: _read_hdf5_node(handle[key], handle, depth=0, max_depth=8) for key in handle.keys()}
            return py, warnings
        except Exception as exc:
            warnings.append(f"Could not load HDF5 paqanalysis file {path.name}: {exc}")
            return None, warnings
    warnings.append(f"Unknown paqanalysis backend for {path.name}")
    return None, warnings


def _extract_hdf5_stim_events(
    path: Path,
    block_index: int,
    frames_per_file: Optional[np.ndarray],
    nplanes: int,
    frame_rate_hz: Optional[float],
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any], List[str]]:
    warnings: List[str] = []
    if h5py is None:
        warnings.append("h5py is unavailable; cannot parse HDF5 paqanalysis file.")
        return None, {}, warnings

    def _decode_ref_char(ref: h5py.Reference, handle: Any) -> str:
        target = handle[ref]
        return _decode_hdf5_char(target)

    def _extract_numeric_series(node: Any, handle: Any) -> np.ndarray:
        if isinstance(node, h5py.Dataset):
            arr = np.asarray(node[()])
            if arr.dtype == h5py.ref_dtype or arr.dtype.kind == "O":
                parts = []
                for item in arr.reshape(-1):
                    if isinstance(item, h5py.Reference):
                        target = handle[item]
                        if isinstance(target, h5py.Dataset):
                            parts.append(np.asarray(target[()]).reshape(-1))
                if parts:
                    return np.concatenate(parts).astype(np.float64)
                return np.array([], dtype=np.float64)
            return arr.reshape(-1).astype(np.float64)
        return np.array([], dtype=np.float64)

    try:
        with h5py.File(path, "r") as handle:
            if "pa" not in handle:
                warnings.append(f"{path.name}: missing top-level pa struct.")
                return None, {}, warnings
            pa = handle["pa"]
            if "frames" not in pa or "stims" not in pa:
                warnings.append(f"{path.name}: pa struct missing frames or stims.")
                return None, {}, warnings

            sample_rate = float(np.asarray(pa["sample_rate"][()]).reshape(-1)[0]) if "sample_rate" in pa else np.nan
            frame_samples = _extract_numeric_series(pa["frames"]["samples"], handle)
            frame_sec = _extract_numeric_series(pa["frames"]["sec"], handle)
            frame_dsamples = float(np.asarray(pa["frames"]["dsamples"][()]).reshape(-1)[0]) if "dsamples" in pa["frames"] else np.nan
            pa_frame_rate = float(np.asarray(pa["frames"]["rate"][()]).reshape(-1)[0]) if "rate" in pa["frames"] else np.nan

            channel_names = []
            for ref in np.asarray(pa["stims"]["chan_name"][()]).reshape(-1):
                if isinstance(ref, h5py.Reference):
                    channel_names.append(_decode_ref_char(ref, handle))
                else:
                    channel_names.append(str(ref))

            block_offsets = np.cumsum(np.r_[0, np.asarray(frames_per_file, dtype=int)[:-1]]) if frames_per_file is not None else np.array([0], dtype=int)
            block_offset = int(block_offsets[block_index]) if block_index < len(block_offsets) else int(block_offsets[-1])

            in_refs = np.asarray(pa["stims"]["in"][()]).reshape(-1)
            out_refs = np.asarray(pa["stims"]["out"][()]).reshape(-1) if "out" in pa["stims"] else np.array([], dtype=object)
            vars_refs = np.asarray(pa["stims"]["vars"][()]).reshape(-1) if "vars" in pa["stims"] else np.array([], dtype=object)
            event_rows: List[Dict[str, Any]] = []
            for idx, item in enumerate(in_refs):
                if not isinstance(item, h5py.Reference):
                    continue
                target_in = handle[item]
                target_out = handle[out_refs[idx]] if idx < out_refs.size and isinstance(out_refs[idx], h5py.Reference) else None
                if not isinstance(target_in, h5py.Group) or "frames" not in target_in:
                    continue
                in_frames_cells = _read_hdf5_node(target_in["frames"], handle, depth=0, max_depth=8) if isinstance(target_in["frames"], h5py.Dataset) else []
                in_sec_cells = _read_hdf5_node(target_in["sec"], handle, depth=0, max_depth=8) if "sec" in target_in and isinstance(target_in["sec"], h5py.Dataset) else []
                in_samples_cells = _read_hdf5_node(target_in["samples"], handle, depth=0, max_depth=8) if "samples" in target_in and isinstance(target_in["samples"], h5py.Dataset) else []
                out_frames_cells = _read_hdf5_node(target_out["frames"], handle, depth=0, max_depth=8) if isinstance(target_out, h5py.Group) and "frames" in target_out and isinstance(target_out["frames"], h5py.Dataset) else []
                out_sec_cells = _read_hdf5_node(target_out["sec"], handle, depth=0, max_depth=8) if isinstance(target_out, h5py.Group) and "sec" in target_out and isinstance(target_out["sec"], h5py.Dataset) else []
                out_samples_cells = _read_hdf5_node(target_out["samples"], handle, depth=0, max_depth=8) if isinstance(target_out, h5py.Group) and "samples" in target_out and isinstance(target_out["samples"], h5py.Dataset) else []
                var_values = _decode_hdf5_ref_item(vars_refs[idx], handle) if idx < vars_refs.size and isinstance(vars_refs[idx], h5py.Reference) else []
                match = re.search(r"(\d+)$", channel_names[idx])
                stim_type = int(match.group(1)) if match else None
                n_vars = max(len(in_frames_cells), len(out_frames_cells), len(var_values) if isinstance(var_values, list) else 0)
                for var_pos in range(n_vars):
                    out_frames = np.asarray(out_frames_cells[var_pos], dtype=np.float64).reshape(-1) if var_pos < len(out_frames_cells) else np.array([], dtype=np.float64)
                    in_frames = np.asarray(in_frames_cells[var_pos], dtype=np.float64).reshape(-1) if var_pos < len(in_frames_cells) else np.array([], dtype=np.float64)
                    use_out = out_frames.size > 0
                    stim_frames_raw = out_frames if use_out else in_frames
                    if stim_frames_raw.size == 0:
                        continue
                    stim_sec = (
                        np.asarray(out_sec_cells[var_pos], dtype=np.float64).reshape(-1)
                        if use_out and var_pos < len(out_sec_cells)
                        else np.asarray(in_sec_cells[var_pos], dtype=np.float64).reshape(-1)
                        if var_pos < len(in_sec_cells)
                        else np.array([], dtype=np.float64)
                    )
                    stim_samples = (
                        np.asarray(out_samples_cells[var_pos], dtype=np.float64).reshape(-1)
                        if use_out and var_pos < len(out_samples_cells)
                        else np.asarray(in_samples_cells[var_pos], dtype=np.float64).reshape(-1)
                        if var_pos < len(in_samples_cells)
                        else np.array([], dtype=np.float64)
                    )
                    local_suite2p_frame = np.round(stim_frames_raw / max(int(nplanes), 1)).astype(int)
                    global_suite2p_frame = local_suite2p_frame + block_offset
                    if frame_rate_hz is not None:
                        stim_time = global_suite2p_frame.astype(np.float64) / frame_rate_hz
                    elif stim_sec.size == global_suite2p_frame.size:
                        stim_time = stim_sec.astype(np.float64)
                        warnings.append(f"{path.name}: used pa.stims sec values because imaging frame rate was unavailable.")
                    else:
                        stim_time = global_suite2p_frame.astype(np.float64)
                        warnings.append(f"{path.name}: stim times recorded as frame indices because no frame rate was available.")

                    stim_var = var_pos + 1
                    program_id = stim_type * 100 + stim_var if stim_type is not None else None
                    for event_idx in range(global_suite2p_frame.size):
                        event_rows.append(
                            {
                                "program_id": program_id,
                                "stim_type": stim_type,
                                "stim_var": stim_var,
                                "stim_channel_name": channel_names[idx],
                                "stim_frame_raw_paq": float(stim_frames_raw[event_idx]),
                                "stim_frame_local_suite2p": int(local_suite2p_frame[event_idx]),
                                "stim_frame_global_suite2p": int(global_suite2p_frame[event_idx]),
                                "stimulation_time": float(stim_time[event_idx]),
                                "stim_sec_paq": float(stim_sec[event_idx]) if stim_sec.size > event_idx else np.nan,
                                "stim_samples_paq": float(stim_samples[event_idx]) if stim_samples.size > event_idx else np.nan,
                                "trigger_source": "out" if use_out else "in",
                            }
                        )

            meta = {
                "sample_rate_hz": sample_rate,
                "pa_frame_rate_hz": pa_frame_rate,
                "pa_frame_samples_per_plane_cycle": frame_dsamples,
                "n_frame_pulses": int(frame_samples.size),
                "n_active_stim_channels": int(len(pd.DataFrame(event_rows)["stim_channel_name"].unique())) if event_rows else 0,
            }
            if not event_rows:
                warnings.append(f"{path.name}: no active stimulation channels were found in pa.stims.in.")
                return None, meta, warnings
            event_df = pd.DataFrame(event_rows).sort_values(["stimulation_time", "program_id"]).reset_index(drop=True)
            return event_df, meta, warnings
    except Exception as exc:
        warnings.append(f"{path.name}: HDF5 paq parsing failed: {exc}")
        return None, {}, warnings


def _resolve_stim_times_and_frames(
    stim_arr: np.ndarray,
    frame_arr: Optional[np.ndarray],
    n_frames: int,
    frame_rate_hz: Optional[float],
    daq_sample_rate_hz: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    warnings: List[str] = []
    stim = _normalize_event_candidates(stim_arr).astype(np.float64)
    if stim.size == 0:
        return stim, stim.astype(int), "empty", warnings

    if frame_arr is not None:
        frame_ref = _normalize_event_candidates(frame_arr).astype(np.float64)
    else:
        frame_ref = np.array([], dtype=np.float64)

    if frame_ref.size >= n_frames:
        frame_ref = frame_ref[:n_frames]
    if frame_ref.size >= 2 and np.all(np.diff(frame_ref) >= 0) and stim.min() >= frame_ref.min() and stim.max() <= frame_ref.max():
        frame_idx = np.interp(stim, frame_ref, np.arange(frame_ref.size, dtype=np.float64))
        if daq_sample_rate_hz is not None and frame_ref.max() > 5.0 * max(frame_ref.size, 1):
            stim_seconds = stim / daq_sample_rate_hz
            method = "interp_stim_samples_against_frame_samples"
        elif frame_rate_hz is not None:
            stim_seconds = frame_idx / frame_rate_hz
            method = "interp_stim_units_onto_frame_index"
        else:
            stim_seconds = frame_idx.astype(np.float64)
            method = "interp_stim_to_frame_index_without_seconds"
            warnings.append("Stim times mapped to frame indices, but no frame rate was available for seconds.")
        return stim_seconds, np.round(frame_idx).astype(int), method, warnings

    if stim.max() <= (n_frames - 1) * 1.2:
        frame_idx = np.round(stim).astype(int)
        if frame_rate_hz is not None:
            stim_seconds = frame_idx / frame_rate_hz
            method = "stim_values_treated_as_frame_indices"
        else:
            stim_seconds = stim.astype(np.float64)
            method = "stim_values_treated_as_frame_indices_without_frame_rate"
            warnings.append("Stim values looked like frame indices, but frame rate was missing.")
        return stim_seconds, frame_idx, method, warnings

    if daq_sample_rate_hz is not None:
        stim_seconds = stim / daq_sample_rate_hz
        if frame_rate_hz is not None:
            frame_idx = np.round(stim_seconds * frame_rate_hz).astype(int)
            method = "stim_values_treated_as_daq_samples"
            return stim_seconds, frame_idx, method, warnings
        warnings.append("Stim values were treated as DAQ samples, but no imaging frame rate was available.")
        return stim_seconds, np.round(stim_seconds).astype(int), "stim_values_treated_as_daq_samples_no_frame_rate", warnings

    if frame_rate_hz is not None and stim.max() <= max(n_frames / frame_rate_hz, 1.0) * 1.5:
        stim_seconds = stim
        frame_idx = np.round(stim * frame_rate_hz).astype(int)
        return stim_seconds, frame_idx, "stim_values_treated_as_seconds", warnings

    warnings.append("Could not confidently infer stim time units; assuming frame indices as least-assumption fallback.")
    return stim, np.round(stim).astype(int), "fallback_assume_frame_indices", warnings


def _compute_window_means(
    traces: np.ndarray,
    stim_frame_idx: np.ndarray,
    frame_rate_hz: float,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], List[str]]:
    warnings: List[str] = []
    baseline_offsets = (int(round(WINDOWS["baseline"][0] * frame_rate_hz)), int(round(WINDOWS["baseline"][1] * frame_rate_hz)))
    stim_offsets = (int(round(WINDOWS["stim"][0] * frame_rate_hz)), int(round(WINDOWS["stim"][1] * frame_rate_hz)))
    post_offsets = (int(round(WINDOWS["post"][0] * frame_rate_hz)), int(round(WINDOWS["post"][1] * frame_rate_hz)))

    n_neurons, n_frames = traces.shape
    baseline_rows: List[np.ndarray] = []
    stim_rows: List[np.ndarray] = []
    post_rows: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []

    for trial_idx, stim_frame in enumerate(stim_frame_idx):
        base_start = stim_frame + baseline_offsets[0]
        base_end = stim_frame + baseline_offsets[1]
        stim_start = stim_frame + stim_offsets[0]
        stim_end = stim_frame + stim_offsets[1]
        post_start = stim_frame + post_offsets[0]
        post_end = stim_frame + post_offsets[1]

        quality_flags: List[str] = []
        if base_start < 0 or base_end > n_frames:
            quality_flags.append("baseline_out_of_bounds")
        if stim_start < 0 or stim_end > n_frames:
            quality_flags.append("stim_out_of_bounds")
        if post_start < 0 or post_end > n_frames:
            quality_flags.append("post_out_of_bounds")

        if quality_flags:
            records.append(
                {
                    "trial_local_index": trial_idx,
                    "stim_frame": int(stim_frame),
                    "quality_flags": "|".join(quality_flags),
                    "is_valid": False,
                }
            )
            baseline_rows.append(np.full(n_neurons, np.nan, dtype=np.float32))
            stim_rows.append(np.full(n_neurons, np.nan, dtype=np.float32))
            post_rows.append(np.full(n_neurons, np.nan, dtype=np.float32))
            continue

        baseline_rows.append(np.nanmean(traces[:, base_start:base_end], axis=1).astype(np.float32))
        stim_rows.append(np.nanmean(traces[:, stim_start:stim_end], axis=1).astype(np.float32))
        post_rows.append(np.nanmean(traces[:, post_start:post_end], axis=1).astype(np.float32))
        records.append(
            {
                "trial_local_index": trial_idx,
                "stim_frame": int(stim_frame),
                "quality_flags": "",
                "is_valid": True,
            }
        )

    baseline = np.vstack(baseline_rows).astype(np.float32)
    stim = np.vstack(stim_rows).astype(np.float32)
    post = np.vstack(post_rows).astype(np.float32)
    diff = stim - baseline
    summary_df = pd.DataFrame(records)
    return summary_df, {"baseline": baseline, "stim": stim, "post": post, "diff": diff}, warnings


def build_trials(
    data_root: str | Path,
    manifest: Dict[str, Any],
    baseline_window: Tuple[float, float] = WINDOWS["baseline"],
    stim_window: Tuple[float, float] = WINDOWS["stim"],
    post_window: Tuple[float, float] = WINDOWS["post"],
) -> Tuple[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]:
    del baseline_window, stim_window, post_window  # windows are kept centralized in WINDOWS
    root = Path(data_root).expanduser().resolve()

    all_trials: List[pd.DataFrame] = []
    session_payloads: Dict[str, Any] = {}
    trial_inference_rows: List[Dict[str, Any]] = []

    for session in manifest["sessions"]:
        session_id = session["session_id"]
        session_path = root / session_id
        LOGGER.info("Building trials for %s", session_id)
        warnings: List[str] = []

        fall_path = session_path / "Fall.mat"
        if not fall_path.exists():
            warnings.append("Missing Fall.mat; skipping session.")
            session_payloads[session_id] = {"warnings": warnings, "used": False}
            continue

        try:
            traces, trace_key = _get_trace_matrix(fall_path)
            iscell, stat_list, ops = _load_suite2p_metadata(fall_path)
        except Exception as exc:
            warnings.append(f"Could not load Suite2p outputs for {session_id}: {exc}")
            session_payloads[session_id] = {"warnings": warnings, "used": False}
            continue

        if iscell.ndim == 1:
            iscell_mask = iscell.astype(np.float64) > 0.5
        else:
            iscell_mask = iscell[:, 0].astype(np.float64) > 0.5
        traces = np.asarray(traces, dtype=np.float32)
        if traces.ndim != 2:
            warnings.append(f"Trace matrix for {session_id} is not 2D: shape={traces.shape}")
            session_payloads[session_id] = {"warnings": warnings, "used": False}
            continue
        if traces.shape[0] != iscell_mask.size and traces.shape[1] == iscell_mask.size:
            traces = traces.T
        if traces.shape[0] != iscell_mask.size:
            warnings.append(
                f"Could not align trace matrix with iscell mask in {session_id}: trace shape={traces.shape}, n_iscell={iscell_mask.size}"
            )
            session_payloads[session_id] = {"warnings": warnings, "used": False}
            continue

        traces = traces[iscell_mask]
        roi_lookup = _roi_centers_from_stat(stat_list)
        roi_lookup = roi_lookup.loc[iscell_mask].reset_index(drop=True)
        roi_lookup["session_id"] = session_id
        roi_lookup["roi_original_index"] = np.flatnonzero(iscell_mask)
        roi_lookup["roi_filtered_index"] = np.arange(len(roi_lookup))

        behavior_files = sorted(session_path.glob("targets/*BhvTraining*VarFile*.mat"))
        vf = None
        if behavior_files:
            try:
                vf = _load_behavior_struct(behavior_files[0])
            except Exception as exc:
                warnings.append(f"Could not load behavior file {behavior_files[0].name}: {exc}")

        frame_rate_hz, frame_rate_warnings = _extract_frame_rate_hz(ops, vf)
        warnings.extend(frame_rate_warnings)
        daq_sample_rate_hz = _extract_daq_sample_rate_hz(vf)
        frames_per_file = np.asarray(ops.get("frames_per_file", []), dtype=int).reshape(-1) if isinstance(ops, dict) else np.array([], dtype=int)
        nplanes = int(ops.get("nplanes", 1)) if isinstance(ops, dict) else 1
        target_catalog, targeted_roi_by_program, target_warnings = _build_target_catalog(session_path, roi_lookup, vf=vf)
        warnings.extend(target_warnings)

        if frame_rate_hz is None:
            warnings.append(f"No usable frame rate for {session_id}; skipping trial extraction.")
            session_payloads[session_id] = {
                "warnings": warnings,
                "used": False,
                "trace_key": trace_key,
                "n_neurons": int(traces.shape[0]),
            }
            continue

        paq_files = sorted(session_path.glob("*_paqanalysis.mat"))
        if not paq_files:
            warnings.append("No paqanalysis files found.")
            session_payloads[session_id] = {"warnings": warnings, "used": False}
            continue

        session_trials: List[pd.DataFrame] = []
        session_arrays: Dict[str, List[np.ndarray]] = {"baseline": [], "stim": [], "post": [], "diff": []}
        inference_objects: List[TrialInference] = []

        for paq_file_index, paq_path in enumerate(paq_files):
            paq_suffix = _extract_numeric_suffix(paq_path, "paqanalysis")
            trial_program_ids = None
            program_mapping_method = "no_program_mapping"
            stim_seconds = None
            stim_frames = None
            hdf5_meta: Dict[str, Any] = {}
            stim_path = None
            frame_path = None
            time_method = "unknown"
            time_warnings: List[str] = []

            if detect_mat_backend(paq_path)["backend"] == "mat_v7.3_hdf5" and h5py is not None:
                event_df, hdf5_meta, paq_warnings = _extract_hdf5_stim_events(
                    path=paq_path,
                    block_index=paq_file_index,
                    frames_per_file=frames_per_file if frames_per_file.size else None,
                    nplanes=nplanes,
                    frame_rate_hz=frame_rate_hz,
                )
                warnings.extend(paq_warnings)
                time_warnings = list(paq_warnings)
                if event_df is None or event_df.empty:
                    continue
                stim_seconds = event_df["stimulation_time"].to_numpy(dtype=np.float64)
                stim_frames = event_df["stim_frame_global_suite2p"].to_numpy(dtype=int)
                trial_program_ids = event_df["program_id"].to_numpy(dtype=float)
                trial_stim_type = event_df["stim_type"].to_numpy(dtype=float) if "stim_type" in event_df else np.full(len(event_df), np.nan)
                trial_stim_var = event_df["stim_var"].to_numpy(dtype=float) if "stim_var" in event_df else np.full(len(event_df), np.nan)
                stim_path = "pa/stims/in"
                frame_path = "pa/frames/samples"
                time_method = "paq_struct_stim_frames_divided_by_nplanes"
                program_mapping_method = "from_stim_channel_suffix"
            else:
                paq_obj, paq_warnings = _load_paqanalysis(paq_path)
                warnings.extend(paq_warnings)
                if paq_obj is None:
                    continue

                flat_arrays = _flatten_numeric_arrays(paq_obj)
                stim_path, stim_arr = _select_candidate_array(flat_arrays, tokens=("stim", "stims", "laser"))
                frame_path, frame_arr = _select_candidate_array(flat_arrays, tokens=("frame", "frames", "paframes"))
                if stim_arr is None:
                    warnings.append(f"{paq_path.name}: could not identify a stimulation event array; skipping file.")
                    continue

                stim_seconds, stim_frames, time_method, time_warnings = _resolve_stim_times_and_frames(
                    stim_arr=stim_arr,
                    frame_arr=frame_arr,
                    n_frames=traces.shape[1],
                    frame_rate_hz=frame_rate_hz,
                    daq_sample_rate_hz=daq_sample_rate_hz,
                )
                warnings.extend([f"{paq_path.name}: {warning}" for warning in time_warnings])
                if stim_frames.size == 0:
                    warnings.append(f"{paq_path.name}: stimulation array was empty after normalization.")
                    continue

                available_program_ids = sorted(int(x) for x in target_catalog["program_id"].dropna().astype(int).tolist())
                program_path, program_id_arr = _extract_program_ids_from_paq(flat_arrays, stim_frames.size, available_program_ids)

                if program_id_arr is not None:
                    trial_program_ids = program_id_arr.astype(float)
                    program_mapping_method = f"from_paq_field:{program_path}"
                elif paq_suffix in targeted_roi_by_program:
                    trial_program_ids = np.full(stim_frames.size, paq_suffix, dtype=float)
                    program_mapping_method = "matched_paq_suffix_to_naparm_suffix"
                else:
                    trial_program_ids = np.full(stim_frames.size, np.nan, dtype=float)
                    program_mapping_method = "no_program_mapping"
                trial_stim_type = np.full(stim_frames.size, np.nan)
                trial_stim_var = np.full(stim_frames.size, np.nan)

            summary_df, arrays_dict, summary_warnings = _compute_window_means(traces, stim_frames, frame_rate_hz)
            warnings.extend([f"{paq_path.name}: {warning}" for warning in summary_warnings])

            trial_df = summary_df.copy()
            trial_df["session_id"] = session_id
            trial_df["paq_file"] = paq_path.name
            trial_df["stimulation_time"] = stim_seconds
            trial_df["trial_id"] = [f"{session_id}__{paq_path.stem}__trial_{idx:04d}" for idx in range(len(trial_df))]
            trial_df["program_id"] = trial_program_ids
            trial_df["stim_type"] = trial_stim_type
            trial_df["stim_var"] = trial_stim_var
            trial_df["block_id"] = paq_suffix
            trial_df["frame_rate_hz"] = frame_rate_hz
            trial_df["trace_key"] = trace_key
            trial_df["t"] = np.arange(len(trial_df), dtype=int)
            if hdf5_meta:
                for key, value in hdf5_meta.items():
                    trial_df[f"paq_{key}"] = value

            dose_map = {
                row["program_id"]: row["dose"]
                for _, row in target_catalog.dropna(subset=["program_id"]).iterrows()
                if pd.notna(row["program_id"])
            }
            trial_df["dose"] = trial_df["program_id"].map(dose_map).astype(float)
            trial_df["condition"] = trial_df["program_id"].apply(
                lambda x: (
                    f"stim{int(x)//100}_var{int(x)%100}"
                    if pd.notna(x) and int(x) >= 100
                    else f"program_{int(x):03d}"
                    if pd.notna(x)
                    else f"block_{paq_suffix:03d}" if paq_suffix is not None else "unknown"
                )
            )

            inference_objects.append(
                TrialInference(
                    session_id=session_id,
                    paq_file=paq_path.name,
                    stim_source_path=stim_path,
                    frame_source_path=frame_path,
                    method="stim_onsets_define_trials",
                    time_unit_assumption=time_method,
                    frame_rate_hz=frame_rate_hz,
                    daq_sample_rate_hz=float(hdf5_meta.get("sample_rate_hz", daq_sample_rate_hz)) if (hdf5_meta or daq_sample_rate_hz is not None) else None,
                    program_mapping_method=program_mapping_method,
                    warnings=list(time_warnings),
                )
            )

            session_trials.append(trial_df)
            for key in session_arrays:
                session_arrays[key].append(arrays_dict[key])

        if not session_trials:
            warnings.append("No usable trials were extracted from paqanalysis files.")
            session_payloads[session_id] = {
                "warnings": warnings,
                "used": False,
                "trace_key": trace_key,
                "n_neurons": int(traces.shape[0]),
            }
            continue

        session_trial_df = pd.concat(session_trials, axis=0, ignore_index=True)
        session_trial_df["trial_order_within_session"] = np.arange(len(session_trial_df), dtype=int)
        session_trial_df["t"] = session_trial_df["trial_order_within_session"]

        dose_vals = session_trial_df["dose"].to_numpy(dtype=np.float64)
        session_trial_df["dose_bucket"] = bucketize_dose(dose_vals)
        session_trial_df["trial_quality_flags"] = session_trial_df["quality_flags"].replace("", np.nan)

        inference_rows = [asdict(item) for item in inference_objects]
        trial_inference_rows.extend(inference_rows)

        session_payloads[session_id] = {
            "used": True,
            "warnings": warnings,
            "trace_key": trace_key,
            "n_frames": int(traces.shape[1]),
            "n_neurons": int(traces.shape[0]),
            "frame_rate_hz": frame_rate_hz,
            "daq_sample_rate_hz": daq_sample_rate_hz,
            "roi_lookup": roi_lookup,
            "target_catalog": target_catalog,
            "targeted_roi_by_program": targeted_roi_by_program,
            "arrays": {key: np.vstack(value).astype(np.float32) for key, value in session_arrays.items()},
            "trial_table": session_trial_df.copy(),
            "trial_inference": inference_rows,
        }
        all_trials.append(session_trial_df)

    trials_df = pd.concat(all_trials, axis=0, ignore_index=True) if all_trials else pd.DataFrame()
    return trials_df, session_payloads, trial_inference_rows


def bucketize_dose(dose_values: np.ndarray) -> List[str]:
    dose = np.asarray(dose_values, dtype=np.float64)
    out = np.full(dose.shape, "unknown", dtype=object)
    finite = np.isfinite(dose)
    if not finite.any():
        return out.tolist()
    unique_vals = np.unique(dose[finite])
    if unique_vals.size == 1:
        out[finite] = "single"
        return out.tolist()
    if unique_vals.size == 2:
        low, high = np.min(unique_vals), np.max(unique_vals)
        out[finite & (dose == low)] = "small"
        out[finite & (dose == high)] = "large"
        return out.tolist()
    q1, q2 = np.quantile(dose[finite], [1.0 / 3.0, 2.0 / 3.0])
    out[finite & (dose <= q1)] = "small"
    out[finite & (dose > q1) & (dose <= q2)] = "medium"
    out[finite & (dose > q2)] = "large"
    return out.tolist()


def _chronological_split_indices(n_rows: int, train_fraction: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
    if n_rows < 4:
        train_n = max(1, n_rows - 1)
    else:
        train_n = min(max(int(math.floor(train_fraction * n_rows)), 3), n_rows - 1)
    train_idx = np.arange(train_n, dtype=int)
    test_idx = np.arange(train_n, n_rows, dtype=int)
    return train_idx, test_idx


def _compute_responsiveness_scores(
    baseline: np.ndarray,
    stim: np.ndarray,
) -> np.ndarray:
    diff = stim - baseline
    mean_effect = np.nanmean(diff, axis=0)
    noise = np.nanstd(diff, axis=0)
    noise = np.where(noise < 1e-6, 1e-6, noise)
    return mean_effect / noise


def select_neurons(
    trials_df: pd.DataFrame,
    neural_data: Dict[str, Any],
    d: int = 10,
    seed: int = 0,
    selection_mode: str = "topk_responsive",
    exclude_targeted_neurons: bool = True,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    selections: Dict[str, Any] = {}
    if trials_df.empty:
        return selections

    for session_id, payload in neural_data.items():
        if not payload.get("used"):
            continue
        session_trials = payload["trial_table"].reset_index(drop=True)
        arrays = payload["arrays"]
        train_idx, test_idx = _chronological_split_indices(len(session_trials))

        baseline_train = arrays["baseline"][train_idx]
        stim_train = arrays["stim"][train_idx]
        scores = _compute_responsiveness_scores(baseline_train, stim_train)
        mean_effect = np.nanmean(stim_train - baseline_train, axis=0)

        candidate_mask = np.isfinite(scores) & np.isfinite(mean_effect)
        if exclude_targeted_neurons:
            targeted_any = np.zeros(scores.shape[0], dtype=bool)
            for program_id, roi_indices in payload.get("targeted_roi_by_program", {}).items():
                del program_id
                for idx in roi_indices:
                    if 0 <= idx < targeted_any.size:
                        targeted_any[idx] = True
            candidate_mask &= ~targeted_any
        responsive_mask = candidate_mask & (mean_effect > 0.0)
        candidate_indices = np.flatnonzero(responsive_mask)
        if candidate_indices.size < d:
            candidate_indices = np.flatnonzero(candidate_mask)

        if candidate_indices.size == 0:
            selections[session_id] = {
                "selected_indices": np.array([], dtype=int),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "warnings": ["No candidate neurons passed responsiveness and metadata filters."],
            }
            continue
        if candidate_indices.size < d:
            selections[session_id] = {
                "selected_indices": np.array([], dtype=int),
                "train_idx": train_idx,
                "test_idx": test_idx,
                "warnings": [f"Session has only {candidate_indices.size} eligible neurons, fewer than requested d={d}."],
            }
            continue

        if selection_mode == "responsive_random":
            selected = np.sort(rng.choice(candidate_indices, size=d, replace=False))
        else:
            order = np.argsort(scores[candidate_indices])[::-1]
            selected = np.sort(candidate_indices[order[:d]])

        selections[session_id] = {
            "selected_indices": selected.astype(int),
            "train_idx": train_idx,
            "test_idx": test_idx,
            "responsiveness_score": scores[selected].astype(float).tolist(),
            "mean_effect": mean_effect[selected].astype(float).tolist(),
            "selection_mode": selection_mode,
            "excluded_targeted_neurons": bool(exclude_targeted_neurons),
            "warnings": [],
        }
    return selections


def _winsorize_train_apply_test(train_x: np.ndarray, test_x: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    lower = np.nanquantile(train_x, lower_q, axis=0)
    upper = np.nanquantile(train_x, upper_q, axis=0)
    train_clip = np.clip(train_x, lower, upper)
    test_clip = np.clip(test_x, lower, upper)
    bounds = pd.DataFrame({"lower": lower.astype(float), "upper": upper.astype(float)})
    return train_clip, test_clip, bounds


def _subtract_train_condition_means(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_conditions: Sequence[Any],
    test_conditions: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[float]]]:
    train_x = np.asarray(train_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)
    train_out = train_x.copy()
    test_out = test_x.copy()
    global_mean = np.nanmean(train_x, axis=0)
    condition_means: Dict[str, List[float]] = {}

    train_conditions = pd.Series(list(train_conditions), dtype="object")
    test_conditions = pd.Series(list(test_conditions), dtype="object")

    for condition in pd.unique(train_conditions):
        mask = train_conditions == condition
        cond_mean = np.nanmean(train_x[mask.to_numpy()], axis=0)
        train_out[mask.to_numpy()] -= cond_mean
        condition_means[str(condition)] = cond_mean.astype(float).tolist()

    for condition in pd.unique(test_conditions):
        mask = test_conditions == condition
        mean_vec = np.asarray(condition_means.get(str(condition), global_mean.tolist()), dtype=np.float64)
        test_out[mask.to_numpy()] -= mean_vec

    return train_out, test_out, condition_means


def _fit_empirical_cdf(train_x: np.ndarray) -> List[Dict[str, np.ndarray]]:
    mappings: List[Dict[str, np.ndarray]] = []
    for col in range(train_x.shape[1]):
        values = np.asarray(train_x[:, col], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            mappings.append(
                {
                    "sorted_values": np.array([0.0], dtype=np.float64),
                    "ecdf_values": np.array([0.5], dtype=np.float64),
                    "n_total": 1,
                }
            )
            continue
        sorter = np.sort(values)
        ecdf = np.arange(1, sorter.size + 1, dtype=np.float64) / (sorter.size + 1.0)
        uniq_values, first_idx = np.unique(sorter, return_index=True)
        uniq_ecdf = np.array([ecdf[idx : next_idx].mean() for idx, next_idx in zip(first_idx, list(first_idx[1:]) + [sorter.size])])
        mappings.append({"sorted_values": uniq_values, "ecdf_values": uniq_ecdf, "n_total": int(sorter.size)})
    return mappings


def _apply_empirical_cdf(x: np.ndarray, mappings: List[Dict[str, np.ndarray]]) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float64)
    for col, mapping in enumerate(mappings):
        values = mapping["sorted_values"]
        ecdf = mapping["ecdf_values"]
        n_total = int(mapping.get("n_total", len(values)))
        if values.size == 1:
            out[:, col] = ecdf[0]
            continue
        left = 1.0 / (n_total + 1.0)
        right = n_total / (n_total + 1.0)
        out[:, col] = np.interp(x[:, col], values, ecdf, left=left, right=right)
    return np.clip(out, 1e-6, 1.0 - 1e-6)


def preprocess_to_copula_table(
    trials_df: pd.DataFrame,
    neural_data: Dict[str, Any],
    d: int = 10,
    seed: int = 0,
    selection_mode: str = "topk_responsive",
    exclude_targeted_neurons: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    selections = select_neurons(
        trials_df=trials_df,
        neural_data=neural_data,
        d=d,
        seed=seed,
        selection_mode=selection_mode,
        exclude_targeted_neurons=exclude_targeted_neurons,
    )

    copula_rows: List[pd.DataFrame] = []
    raw_rows: List[pd.DataFrame] = []
    lookup_rows: List[Dict[str, Any]] = []
    preprocess_meta: Dict[str, Any] = {
        "selection_mode": selection_mode,
        "seed": seed,
        "d": d,
        "exclude_targeted_neurons": exclude_targeted_neurons,
        "sessions": {},
        "assumptions": [
            "Analysis slice = session_id. This keeps neuron identity stable within each fitted model and uses chronological trial order as time index t.",
            "Neuron selection is performed on training trials only to avoid leakage.",
            "Within each dose bucket, train means are subtracted from both train and test trials to remove gross condition shifts before dependence modeling.",
            "Outliers are handled by 1%/99% winsorization using train quantiles only.",
            "Pseudo-observations are empirical-CDF transforms fit on train only and applied to test by interpolation.",
        ],
    }

    for session_id, selection in selections.items():
        payload = neural_data[session_id]
        selected = np.asarray(selection["selected_indices"], dtype=int)
        if selected.size == 0:
            preprocess_meta["sessions"][session_id] = selection
            continue

        session_trials = payload["trial_table"].reset_index(drop=True)
        arrays = payload["arrays"]
        train_idx = np.asarray(selection["train_idx"], dtype=int)
        test_idx = np.asarray(selection["test_idx"], dtype=int)

        raw_diff = arrays["diff"][:, selected].astype(np.float64)
        train_raw = raw_diff[train_idx]
        test_raw = raw_diff[test_idx]
        train_conditions = session_trials.loc[train_idx, "dose_bucket"].fillna("unknown").tolist()
        test_conditions = session_trials.loc[test_idx, "dose_bucket"].fillna("unknown").tolist()

        train_centered, test_centered, condition_means = _subtract_train_condition_means(
            train_x=train_raw,
            test_x=test_raw,
            train_conditions=train_conditions,
            test_conditions=test_conditions,
        )
        train_wins, test_wins, bounds = _winsorize_train_apply_test(train_centered, test_centered)
        mappings = _fit_empirical_cdf(train_wins)

        u_train = _apply_empirical_cdf(train_wins, mappings)
        u_test = _apply_empirical_cdf(test_wins, mappings)

        train_df = session_trials.loc[train_idx].copy()
        test_df = session_trials.loc[test_idx].copy()
        train_df["split"] = "train"
        test_df["split"] = "test"
        combined_meta = pd.concat([train_df, test_df], axis=0, ignore_index=True)

        raw_combined = np.vstack([train_wins, test_wins])
        u_combined = np.vstack([u_train, u_test])

        for col_idx in range(selected.size):
            x_name = f"x{col_idx + 1}"
            raw_name = f"raw_{x_name}"
            combined_meta[x_name] = u_combined[:, col_idx]
            combined_meta[raw_name] = raw_combined[:, col_idx]
            roi_row = payload["roi_lookup"].iloc[selected[col_idx]]
            lookup_rows.append(
                {
                    "session_id": session_id,
                    "analysis_slice_id": session_id,
                    "selection_mode": selection_mode,
                    "x_col": x_name,
                    "roi_filtered_index": int(roi_row["roi_filtered_index"]),
                    "roi_original_index": int(roi_row["roi_original_index"]),
                    "x_center": float(roi_row["x_center"]) if pd.notna(roi_row["x_center"]) else np.nan,
                    "y_center": float(roi_row["y_center"]) if pd.notna(roi_row["y_center"]) else np.nan,
                    "responsiveness_score": float(selection["responsiveness_score"][col_idx]),
                    "mean_effect_train": float(selection["mean_effect"][col_idx]),
                }
            )

        combined_meta["analysis_slice_id"] = session_id
        combined_meta["selection_mode"] = selection_mode
        combined_meta["window_type"] = "stim_minus_baseline"

        raw_cols = [f"raw_x{i + 1}" for i in range(selected.size)]
        x_cols = [f"x{i + 1}" for i in range(selected.size)]
        raw_rows.append(combined_meta[[
            "trial_id",
            "session_id",
            "analysis_slice_id",
            "stimulation_time",
            "t",
            "condition",
            "dose",
            "dose_bucket",
            "split",
            "selection_mode",
            "window_type",
        ] + raw_cols].copy())
        copula_rows.append(combined_meta[[
            "trial_id",
            "session_id",
            "analysis_slice_id",
            "stimulation_time",
            "t",
            "condition",
            "dose",
            "dose_bucket",
            "split",
            "selection_mode",
            "window_type",
        ] + x_cols].copy())

        preprocess_meta["sessions"][session_id] = {
            "selected_indices": selected.astype(int).tolist(),
            "train_n": int(train_idx.size),
            "test_n": int(test_idx.size),
            "condition_means": condition_means,
            "winsor_bounds": bounds.to_dict(orient="records"),
            "selection_mode": selection_mode,
            "excluded_targeted_neurons": exclude_targeted_neurons,
            "warnings": selection.get("warnings", []),
        }

    copula_df = pd.concat(copula_rows, axis=0, ignore_index=True) if copula_rows else pd.DataFrame()
    raw_df = pd.concat(raw_rows, axis=0, ignore_index=True) if raw_rows else pd.DataFrame()
    lookup_df = pd.DataFrame(lookup_rows)
    return copula_df, raw_df, lookup_df, preprocess_meta


def _gaussian_copula_nll_fit_eval(u_train: np.ndarray, u_test: np.ndarray, ridge: float = 1e-4) -> float:
    z_train = norm.ppf(np.clip(np.asarray(u_train, dtype=np.float64), 1e-6, 1.0 - 1e-6))
    z_test = norm.ppf(np.clip(np.asarray(u_test, dtype=np.float64), 1e-6, 1.0 - 1e-6))
    if z_train.shape[0] < 3 or z_test.shape[0] < 1:
        return float("nan")
    corr = np.corrcoef(z_train, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    corr += ridge * np.eye(corr.shape[0])
    dstd = np.sqrt(np.clip(np.diag(corr), 1e-12, None))
    corr = corr / np.outer(dstd, dstd)
    np.fill_diagonal(corr, 1.0)
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0 or not np.isfinite(logdet):
        return float("nan")
    inv_corr = np.linalg.inv(corr)
    quad = np.einsum("ni,ij,nj->n", z_test, inv_corr - np.eye(corr.shape[0]), z_test)
    log_c = -0.5 * logdet - 0.5 * quad
    return float(-np.mean(log_c))


def run_minimal_evaluation(
    out_root: str | Path,
    copula_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out_root = Path(out_root).expanduser().resolve()
    if copula_df is None:
        copula_path = out_root / "copula_ready_table.csv"
        if not copula_path.exists():
            raise FileNotFoundError(f"Missing copula-ready table at {copula_path}")
        copula_df = pd.read_csv(copula_path)

    x_cols = [col for col in copula_df.columns if re.fullmatch(r"x\d+", col)]
    eval_rows: List[Dict[str, Any]] = []
    eval_meta: Dict[str, Any] = {
        "used_repo_vine_functions": False,
        "notes": [],
    }

    repo_helpers = None
    src_root = Path(__file__).resolve().parent / "src"
    if src_root.exists():
        sys.path.insert(0, str(src_root))
        try:
            from dvc_package.experiments.simulation_benchmarks import (  # type: ignore
                _fit_parametric_vine,
                _fit_truncated_cvine_level0,
                _mean_copula_nll,
            )

            repo_helpers = {
                "fit_full": _fit_parametric_vine,
                "fit_trunc": _fit_truncated_cvine_level0,
                "mean_nll": _mean_copula_nll,
            }
            eval_meta["used_repo_vine_functions"] = True
        except Exception as exc:  # pragma: no cover - environment dependent
            eval_meta["notes"].append(f"Repo vine helpers unavailable, falling back to Gaussian-only evaluation: {exc}")

    group_cols = ["analysis_slice_id", "dose_bucket", "selection_mode"]
    for group_key, group in copula_df.groupby(group_cols, dropna=False):
        train = group[group["split"] == "train"]
        test = group[group["split"] == "test"]
        if train.empty or test.empty:
            continue
        group_x_cols = [col for col in x_cols if col in group.columns and not group[col].isna().all()]
        if not group_x_cols:
            continue
        u_train = train[group_x_cols].to_numpy(dtype=np.float64)
        u_test = test[group_x_cols].to_numpy(dtype=np.float64)
        gaussian_nll = _gaussian_copula_nll_fit_eval(u_train, u_test)
        row = {
            "analysis_slice_id": group_key[0],
            "dose_bucket": group_key[1],
            "selection_mode": group_key[2],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "gaussian_copula_nll": gaussian_nll,
            "truncated_vine_nll": np.nan,
            "full_vine_nll": np.nan,
            "TC_higher": np.nan,
            "evaluation_backend": "gaussian_only_fallback",
        }

        if repo_helpers is not None:
            try:
                full_vine = repo_helpers["fit_full"](
                    x_train=u_train.astype(np.float32),
                    families=["ind", "gaussian", "student", "clayton", "frank", "gumbel", "joe"],
                    optimize_structure=True,
                    seed=0,
                )
                trunc_vine = repo_helpers["fit_trunc"](
                    x_train=u_train.astype(np.float32),
                    families=["ind", "gaussian", "student", "clayton", "frank", "gumbel", "joe"],
                    order=list(range(u_train.shape[1])),
                )
                full_nll = repo_helpers["mean_nll"](full_vine, u_test.astype(np.float32))
                trunc_nll = repo_helpers["mean_nll"](trunc_vine, u_test.astype(np.float32))
                row["truncated_vine_nll"] = float(trunc_nll)
                row["full_vine_nll"] = float(full_nll)
                row["TC_higher"] = float(trunc_nll - full_nll)
                row["evaluation_backend"] = "repo_vine_helpers"
            except Exception as exc:  # pragma: no cover - environment dependent
                eval_meta["notes"].append(
                    f"Vine fit failed for slice={group_key[0]}, dose_bucket={group_key[1]}: {exc}"
                )

        eval_rows.append(row)

    eval_df = pd.DataFrame(eval_rows)
    return eval_df, eval_meta


def write_outputs(
    out_root: str | Path,
    manifest: Dict[str, Any],
    trials_df: pd.DataFrame,
    neural_data: Dict[str, Any],
    copula_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    neuron_lookup_df: pd.DataFrame,
    preprocess_meta: Dict[str, Any],
    trial_inference_rows: List[Dict[str, Any]],
    evaluation_df: Optional[pd.DataFrame] = None,
    evaluation_meta: Optional[Dict[str, Any]] = None,
) -> None:
    out_root = Path(out_root).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    write_json(out_root / "dataset_manifest.json", manifest)
    if not trials_df.empty:
        trials_df.to_csv(out_root / "trial_table.csv", index=False)
    if not copula_df.empty:
        copula_df.to_csv(out_root / "copula_ready_table.csv", index=False)
    if not raw_df.empty:
        raw_df.to_csv(out_root / "raw_residual_table.csv", index=False)
    if not neuron_lookup_df.empty:
        neuron_lookup_df.to_csv(out_root / "neuron_lookup.csv", index=False)
    if trial_inference_rows:
        pd.DataFrame(trial_inference_rows).to_csv(out_root / "trial_inference.csv", index=False)

    session_meta = {}
    for session_id, payload in neural_data.items():
        session_meta[session_id] = {
            "used": bool(payload.get("used", False)),
            "warnings": payload.get("warnings", []),
            "trace_key": payload.get("trace_key"),
            "n_neurons": payload.get("n_neurons"),
            "n_frames": payload.get("n_frames"),
            "frame_rate_hz": payload.get("frame_rate_hz"),
            "daq_sample_rate_hz": payload.get("daq_sample_rate_hz"),
            "raw_files_used": {
                "fall_mat": str(Path(manifest["data_root"]) / session_id / "Fall.mat"),
                "paqanalysis": [
                    str(path)
                    for path in sorted((Path(manifest["data_root"]) / session_id).glob("*_paqanalysis.mat"))
                ],
                "targets": [
                    str(path)
                    for path in sorted((Path(manifest["data_root"]) / session_id / "targets").glob("*.mat"))
                ],
            },
        }
    write_json(
        out_root / "analysis_metadata.json",
        {
            "windows_seconds": WINDOWS,
            "sessions": session_meta,
            "preprocessing": preprocess_meta,
            "assumptions": [
                "Stim onsets are the primary trial markers. If explicit trial markers are absent, each stim onset defines one pseudo-trial.",
                "Dose is taken from the number of selected NAPARM target points when a per-trial or per-block program mapping is available.",
                "When a paqanalysis file cannot be read or timed confidently, that file is skipped rather than silently guessed.",
                "Time index t is chronological trial order within session because that is the most stable session-local ordering for DVC.",
            ],
        },
    )
    if evaluation_df is not None and not evaluation_df.empty:
        evaluation_df.to_csv(out_root / "evaluation_summary.csv", index=False)
    if evaluation_meta is not None:
        write_json(out_root / "evaluation_metadata.json", evaluation_meta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a DVC-ready dataset from the Dalgleish stimulation dataset.")
    parser.add_argument("--data_root", required=True, help="Path to the dataset root containing session directories.")
    parser.add_argument("--out_root", default="dvc_ready", help="Output directory.")
    parser.add_argument("--d", type=int, default=10, help="Target number of neurons per session slice.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--selection_mode",
        choices=["responsive_random", "topk_responsive"],
        default="topk_responsive",
        help="Neuron selection mode.",
    )
    parser.add_argument(
        "--include_targeted_neurons",
        action="store_true",
        help="Include directly targeted neurons when target-to-ROI mapping is available.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    LOGGER.info("Phase 1/6: discovering dataset at %s", args.data_root)
    manifest = build_manifest(args.data_root)
    print_manifest_summary(manifest)

    LOGGER.info("Phase 2/6: building trial-aligned summaries")
    trials_df, neural_data, trial_inference_rows = build_trials(args.data_root, manifest)
    if trials_df.empty:
        LOGGER.warning("No usable trials were extracted. Writing manifest and metadata only.")
        write_outputs(
            out_root=args.out_root,
            manifest=manifest,
            trials_df=trials_df,
            neural_data=neural_data,
            copula_df=pd.DataFrame(),
            raw_df=pd.DataFrame(),
            neuron_lookup_df=pd.DataFrame(),
            preprocess_meta={},
            trial_inference_rows=trial_inference_rows,
            evaluation_df=pd.DataFrame(),
            evaluation_meta={"notes": ["No copula-ready table was written because no usable trials were extracted."]},
        )
        return

    LOGGER.info("Phase 3-4/6: selecting neurons and preprocessing without leakage")
    copula_df, raw_df, neuron_lookup_df, preprocess_meta = preprocess_to_copula_table(
        trials_df=trials_df,
        neural_data=neural_data,
        d=args.d,
        seed=args.seed,
        selection_mode=args.selection_mode,
        exclude_targeted_neurons=not args.include_targeted_neurons,
    )
    LOGGER.info("Built copula-ready table with shape %s", tuple(copula_df.shape))
    if not copula_df.empty:
        LOGGER.info("Example output schema: %s", ", ".join(copula_df.columns[: min(len(copula_df.columns), 20)]))

    LOGGER.info("Phase 5/6: running minimal evaluation harness")
    evaluation_df, evaluation_meta = run_minimal_evaluation(args.out_root, copula_df=copula_df)

    LOGGER.info("Phase 6/6: writing outputs to %s", args.out_root)
    write_outputs(
        out_root=args.out_root,
        manifest=manifest,
        trials_df=trials_df,
        neural_data=neural_data,
        copula_df=copula_df,
        raw_df=raw_df,
        neuron_lookup_df=neuron_lookup_df,
        preprocess_meta=preprocess_meta,
        trial_inference_rows=trial_inference_rows,
        evaluation_df=evaluation_df,
        evaluation_meta=evaluation_meta,
    )

    LOGGER.info("Done. Wrote manifest, trial table, copula-ready table, lookup table, metadata, and evaluation outputs.")


if __name__ == "__main__":
    main()
