"""Allen Visual Behavior Neuropixels preprocessing helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd


def decode_bytes_array(values: np.ndarray) -> np.ndarray:
    out = []
    for value in np.asarray(values):
        if isinstance(value, (bytes, np.bytes_)):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return np.asarray(out, dtype=object)


def _sanitize_region_label(label: str) -> str:
    label = str(label).strip()
    if not label or label.lower() == "root":
        return "root"
    return label.replace("-mo", "").replace("-po", "").replace("-sg", "").replace("-sh", "").replace("-co", "")


@dataclass
class AllenVBNSessionSummary:
    session_id: int
    mouse_id: str
    experience_level: str
    stimulus_table_name: str
    n_presentations: int
    n_active_presentations: int
    n_good_units: int
    region_unit_counts: Dict[str, int]


@dataclass
class AllenVBNSessionData:
    session_id: int
    mouse_id: str
    experience_level: str
    stimulus_table_name: str
    presentations: pd.DataFrame
    unit_table: pd.DataFrame
    region_names: List[str]
    presentation_matrix: np.ndarray
    summary: AllenVBNSessionSummary


def _read_dataset(f: h5py.File, key: str) -> np.ndarray:
    arr = f[key][()]
    if isinstance(arr, bytes):
        return np.asarray(arr.decode("utf-8"), dtype=object)
    return np.asarray(arr)


def _load_unit_table(f: h5py.File) -> pd.DataFrame:
    quality = decode_bytes_array(_read_dataset(f, "/units/quality"))
    presence_ratio = _read_dataset(f, "/units/presence_ratio").astype(np.float64)
    firing_rate = _read_dataset(f, "/units/firing_rate").astype(np.float64)
    peak_channel_id = _read_dataset(f, "/units/peak_channel_id").astype(np.int64)
    unit_ids = _read_dataset(f, "/units/id").astype(np.int64)

    electrode_ids = _read_dataset(f, "/general/extracellular_ephys/electrodes/id").astype(np.int64)
    locations = decode_bytes_array(_read_dataset(f, "/general/extracellular_ephys/electrodes/location"))
    region_by_channel = {
        int(channel_id): _sanitize_region_label(region)
        for channel_id, region in zip(electrode_ids, locations)
    }
    region_labels = [region_by_channel.get(int(ch), "unknown") for ch in peak_channel_id]

    spike_index = _read_dataset(f, "/units/spike_times_index").astype(np.int64)
    spikes_per_unit = np.diff(np.concatenate([[0], spike_index]))

    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "quality": quality,
            "presence_ratio": presence_ratio,
            "firing_rate": firing_rate,
            "peak_channel_id": peak_channel_id,
            "region": region_labels,
            "n_spikes": spikes_per_unit,
        }
    )


def _select_stimulus_table_name(f: h5py.File) -> str:
    candidates = []
    for key in f["/intervals"].keys():
        if not key.endswith("_presentations"):
            continue
        group = f[f"/intervals/{key}"]
        n_rows = int(group["start_time"].shape[0]) if "start_time" in group else 0
        candidates.append((n_rows, key))
    if not candidates:
        raise ValueError("No *_presentations stimulus table found in /intervals")
    candidates.sort(reverse=True)
    return str(candidates[0][1])


def _load_presentations(f: h5py.File, table_name: str) -> pd.DataFrame:
    group = f[f"/intervals/{table_name}"]
    data: Dict[str, np.ndarray] = {}
    for key in group.keys():
        if key in {"tags", "tags_index", "timeseries", "timeseries_index"}:
            continue
        arr = _read_dataset(f, f"/intervals/{table_name}/{key}")
        if arr.dtype.kind in {"S", "O", "U"}:
            arr = decode_bytes_array(arr)
        data[key] = np.asarray(arr)
    return pd.DataFrame(data)


def summarize_allen_vbn_session(
    session_path: Path,
    *,
    manifest_path: Optional[Path] = None,
    quality: str = "good",
    min_presence_ratio: float = 0.95,
) -> AllenVBNSessionSummary:
    session_path = Path(session_path)
    with h5py.File(session_path, "r") as f:
        session_id = int(_read_dataset(f, "/identifier"))
        mouse_id = str(_read_dataset(f, "/general/subject/subject_id").item())
        table_name = _select_stimulus_table_name(f)
        presentations = _load_presentations(f, table_name)
        units = _load_unit_table(f)
    manifest_mouse, manifest_exp = _load_session_meta_from_manifest(session_id, manifest_path)
    if manifest_mouse:
        mouse_id = manifest_mouse

    mask = (units["quality"] == quality) & (units["presence_ratio"] >= float(min_presence_ratio))
    units = units.loc[mask].copy()
    units = units[~units["region"].isin({"root", "unknown"})]
    counts = units["region"].value_counts().sort_values(ascending=False)

    if "active" in presentations.columns:
        n_active = int(np.sum(presentations["active"].astype(bool).to_numpy()))
    else:
        n_active = int(len(presentations))

    return AllenVBNSessionSummary(
        session_id=session_id,
        mouse_id=mouse_id,
        experience_level=manifest_exp,
        stimulus_table_name=table_name,
        n_presentations=int(len(presentations)),
        n_active_presentations=n_active,
        n_good_units=int(len(units)),
        region_unit_counts={str(k): int(v) for k, v in counts.items()},
    )


def _load_session_meta_from_manifest(session_id: int, manifest_path: Optional[Path]) -> Tuple[str, str]:
    if manifest_path is None or not manifest_path.exists():
        return "", ""
    payload = json.loads(manifest_path.read_text())
    rows = payload.get("rows")
    if not isinstance(rows, list):
        return "", ""
    for row in rows:
        if int(row["ecephys_session_id"]) == int(session_id):
            return str(row.get("mouse_id", "")), str(row.get("experience_level", ""))
    return "", ""


def extract_region_presentation_matrix(
    session_path: Path,
    *,
    manifest_path: Optional[Path] = None,
    min_presence_ratio: float = 0.95,
    min_firing_rate: float = 0.1,
    top_k_regions: int = 6,
    min_units_per_region: int = 20,
    min_regions: int = 3,
    quality: str = "good",
    response_window: Tuple[float, float] = (0.0, 0.25),
    active_only: bool = True,
    drop_omitted: bool = True,
) -> AllenVBNSessionData:
    session_path = Path(session_path)
    with h5py.File(session_path, "r") as f:
        session_id = int(_read_dataset(f, "/identifier"))
        mouse_id = str(_read_dataset(f, "/general/subject/subject_id").item())
        table_name = _select_stimulus_table_name(f)
        presentations_all = _load_presentations(f, table_name)
        units = _load_unit_table(f)
        spike_times = _read_dataset(f, "/units/spike_times").astype(np.float64)
        spike_index = _read_dataset(f, "/units/spike_times_index").astype(np.int64)

    manifest_mouse, manifest_exp = _load_session_meta_from_manifest(session_id, manifest_path)
    if manifest_mouse:
        mouse_id = manifest_mouse
    experience_level = manifest_exp

    presentations = presentations_all.copy()
    if active_only and "active" in presentations.columns:
        presentations = presentations.loc[presentations["active"].astype(bool)].copy()
    if drop_omitted and "omitted" in presentations.columns:
        presentations = presentations.loc[~presentations["omitted"].astype(bool)].copy()
    presentations = presentations.sort_values("start_time").reset_index(drop=True)

    unit_mask = (
        (units["quality"] == quality)
        & (units["presence_ratio"] >= float(min_presence_ratio))
        & (units["firing_rate"] >= float(min_firing_rate))
        & (~units["region"].isin({"root", "unknown"}))
    )
    units = units.loc[unit_mask].copy().reset_index(drop=True)

    region_counts = units["region"].value_counts().sort_values(ascending=False)
    kept_regions = [
        str(region)
        for region, count in region_counts.items()
        if int(count) >= int(min_units_per_region)
    ][: int(top_k_regions)]
    if len(kept_regions) < int(min_regions):
        raise ValueError(
            f"Need at least {int(min_regions)} regions after filtering; got {len(kept_regions)} "
            f"from {session_path}"
        )
    units = units.loc[units["region"].isin(kept_regions)].copy().reset_index(drop=True)

    region_to_unit_idx: Dict[str, List[int]] = {}
    for idx, region in enumerate(units["region"]):
        region_to_unit_idx.setdefault(str(region), []).append(int(idx))

    starts = presentations["start_time"].to_numpy(dtype=np.float64) + float(response_window[0])
    stops = presentations["start_time"].to_numpy(dtype=np.float64) + float(response_window[1])
    n_presentations = len(presentations)
    matrix = np.zeros((n_presentations, len(kept_regions)), dtype=np.float32)

    spike_starts = np.concatenate([[0], spike_index[:-1]])
    spike_stops = spike_index

    for col_idx, region in enumerate(kept_regions):
        unit_indices = region_to_unit_idx[region]
        region_counts_mat = np.zeros((n_presentations, len(unit_indices)), dtype=np.float32)
        for u_idx_local, unit_idx in enumerate(unit_indices):
            s0 = int(spike_starts[unit_idx])
            s1 = int(spike_stops[unit_idx])
            spikes = spike_times[s0:s1]
            left = np.searchsorted(spikes, starts, side="left")
            right = np.searchsorted(spikes, stops, side="left")
            region_counts_mat[:, u_idx_local] = (right - left).astype(np.float32)
        matrix[:, col_idx] = region_counts_mat.mean(axis=1)

    summary = AllenVBNSessionSummary(
        session_id=session_id,
        mouse_id=mouse_id,
        experience_level=experience_level,
        stimulus_table_name=table_name,
        n_presentations=int(len(presentations_all)),
        n_active_presentations=int(len(presentations)),
        n_good_units=int(len(units)),
        region_unit_counts={str(region): int(region_counts[region]) for region in kept_regions},
    )

    return AllenVBNSessionData(
        session_id=session_id,
        mouse_id=mouse_id,
        experience_level=experience_level,
        stimulus_table_name=table_name,
        presentations=presentations,
        unit_table=units,
        region_names=kept_regions,
        presentation_matrix=matrix,
        summary=summary,
    )


def load_allen_vbn_session(
    session_path: Path,
    *,
    manifest_path: Optional[Path] = None,
    **kwargs: object,
) -> AllenVBNSessionData:
    return extract_region_presentation_matrix(
        session_path,
        manifest_path=manifest_path,
        **kwargs,
    )
