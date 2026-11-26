"""Both established DVC exports and the modulation estimator resolve through the installed namespace."""
import subprocess
import sys


def test_modulation_import_is_independent_of_torch():
    check = (
        "import sys; from dvc_package.modulation import fit, pooled_change; "
        "assert callable(fit) and callable(pooled_change); assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


def test_root_keeps_existing_public_symbols():
    import dvc_package as dvc
    from dvc_package import core
    for name in dvc.__all__:
        assert name in dir(dvc)
        if name != "optimize_vine_structure":
            assert getattr(dvc, name) is getattr(core, name)
    assert callable(dvc.optimize_vine_structure)
