"""
Tests for VDC (vine-diffusion-copula) backend integration.

These tests verify the backend module works correctly both when ``vdc`` is
installed and when it is not.  The majority test graceful degradation since
``vdc`` may not be available in all environments.
"""

import pytest
import numpy as np


class TestVDCAvailability:
    """Test the vdc_available() check and import handling."""

    def test_vdc_available_returns_bool(self):
        from dvc_package.baselines.vdc_backend import vdc_available
        result = vdc_available()
        assert isinstance(result, bool)

    def test_vdc_available_is_idempotent(self):
        from dvc_package.baselines.vdc_backend import vdc_available
        r1 = vdc_available()
        r2 = vdc_available()
        assert r1 == r2


class TestVDCBackendWhenUnavailable:
    """Test VDCBackend behavior when vdc is not installed."""

    def test_constructor_raises_import_error(self):
        from dvc_package.baselines.vdc_backend import vdc_available, VDCBackend
        if vdc_available():
            pytest.skip("vdc is installed - cannot test unavailable path")
        with pytest.raises(ImportError, match="vdc"):
            VDCBackend(model=None)

    def test_from_checkpoint_raises_import_error(self):
        from dvc_package.baselines.vdc_backend import vdc_available, VDCBackend
        if vdc_available():
            pytest.skip("vdc is installed")
        with pytest.raises(ImportError, match="vdc"):
            VDCBackend.from_checkpoint("/nonexistent/model.pt")


class TestVDCBackendInterface:
    """Test VDCBackend interface when vdc IS available (skipped if not)."""

    @pytest.fixture(autouse=True)
    def _require_vdc(self):
        from dvc_package.baselines.vdc_backend import vdc_available
        if not vdc_available():
            pytest.skip("vdc not installed")

    def test_backend_has_expected_methods(self):
        from dvc_package.baselines.vdc_backend import VDCBackend
        # Just check the class has the expected API surface
        for method_name in ("fit", "logpdf", "pdf", "simulate", "rosenblatt",
                            "nll", "save_vine", "load_vine",
                            "evaluate_against_dvc", "from_checkpoint"):
            assert hasattr(VDCBackend, method_name), f"Missing method: {method_name}"

    def test_unfit_backend_raises_on_logpdf(self):
        """Calling logpdf before fit should raise RuntimeError."""
        from dvc_package.baselines.vdc_backend import VDCBackend
        import torch
        # Minimal construction
        from vdc.models.unet_grid import GridUNet
        model = GridUNet(m=32, in_channels=1, base_channels=16,
                         channel_mults=(1, 2), num_res_blocks=1)
        backend = VDCBackend(model=model, device="cpu", m=32)
        with pytest.raises(RuntimeError, match="fit"):
            backend.logpdf(np.random.rand(10, 3))

    def test_unfit_backend_raises_on_simulate(self):
        from dvc_package.baselines.vdc_backend import VDCBackend
        import torch
        from vdc.models.unet_grid import GridUNet
        model = GridUNet(m=32, in_channels=1, base_channels=16,
                         channel_mults=(1, 2), num_res_blocks=1)
        backend = VDCBackend(model=model, device="cpu", m=32)
        with pytest.raises(RuntimeError, match="fit"):
            backend.simulate(100)
