from pathlib import Path

import h5py
import numpy as np

from dvc_package.real_data import extract_region_presentation_matrix


def _bytes(values):
    return np.asarray([v.encode("utf-8") for v in values], dtype="S16")


def _make_minimal_allen_file(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("/identifier", data=np.int64(123456))
        f.create_dataset("/general/subject/subject_id", data=np.bytes_("mouse-a"))

        electrode_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
        locations = _bytes(["VISp", "VISp", "CA1", "LP"])
        f.create_dataset("/general/extracellular_ephys/electrodes/id", data=electrode_ids)
        f.create_dataset("/general/extracellular_ephys/electrodes/location", data=locations)

        f.create_dataset("/units/id", data=np.asarray([100, 101, 102, 103], dtype=np.int64))
        f.create_dataset("/units/quality", data=_bytes(["good", "good", "good", "noise"]))
        f.create_dataset("/units/presence_ratio", data=np.asarray([0.99, 0.97, 0.98, 0.99], dtype=np.float64))
        f.create_dataset("/units/firing_rate", data=np.asarray([5.0, 4.0, 3.0, 10.0], dtype=np.float64))
        f.create_dataset("/units/peak_channel_id", data=np.asarray([10, 11, 12, 13], dtype=np.int64))

        spike_times = np.asarray(
            [
                0.10, 0.22, 0.61, 1.05,  # unit 100
                0.05, 0.24, 0.65, 1.10,  # unit 101
                0.18, 0.55, 0.75, 1.02,  # unit 102
                0.10, 0.11, 0.12,        # unit 103 (noise)
            ],
            dtype=np.float64,
        )
        spike_index = np.asarray([4, 8, 12, 15], dtype=np.int64)
        f.create_dataset("/units/spike_times", data=spike_times)
        f.create_dataset("/units/spike_times_index", data=spike_index)

        grp = f.create_group("/intervals/test_presentations")
        grp.create_dataset("start_time", data=np.asarray([0.0, 0.5, 1.0], dtype=np.float64))
        grp.create_dataset("stop_time", data=np.asarray([0.25, 0.75, 1.25], dtype=np.float64))
        grp.create_dataset("active", data=np.asarray([True, True, True], dtype=bool))
        grp.create_dataset("omitted", data=np.asarray([False, False, False], dtype=bool))
        grp.create_dataset("is_change", data=np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        grp.create_dataset("rewarded", data=np.asarray([0.0, 1.0, 0.0], dtype=np.float64))


def test_extract_region_presentation_matrix(tmp_path: Path):
    path = tmp_path / "minimal_allen.nwb"
    _make_minimal_allen_file(path)

    session = extract_region_presentation_matrix(
        path,
        top_k_regions=3,
        min_units_per_region=1,
        min_regions=2,
        min_presence_ratio=0.95,
        min_firing_rate=0.1,
        response_window=(0.0, 0.25),
    )

    assert session.session_id == 123456
    assert session.region_names == ["VISp", "CA1"]
    assert session.presentation_matrix.shape == (3, 2)
    np.testing.assert_allclose(
        session.presentation_matrix,
        np.asarray(
            [
                [2.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    assert session.summary.region_unit_counts == {"VISp": 2, "CA1": 1}
