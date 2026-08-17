"""Shared test fixtures for alex-translator tests.

Stubs heavy external dependencies (torch/GPU) so pure logic and
HTTP integration can be tested without models installed.
"""

import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


def _install_torch_stub():
    # Si hay un torch real instalado (p.ej. venv compartido con GPU),
    # usarlo — el stub solo aplica cuando no existe ningún torch.
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    if "torch" in sys.modules:
        return
    import importlib.util
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0.0-stub"
    # __spec__ es necesario: pytest llama a importlib.util.find_spec('torch')
    # y falla con ValueError si el módulo en sys.modules no lo tiene.
    torch.__spec__ = importlib.util.spec_from_loader("torch", loader=None)
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.no_grad = lambda: _nullcontext()
    torch.device = lambda *a, **k: "cpu"
    torch.tensor = lambda *a, **k: _TensorStub()
    torch.zeros = lambda *a, **k: _TensorStub()
    sys.modules["torch"] = torch


class _nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


class _TensorStub:
    """Minimal tensor stub for tests (no GPU needed)."""
    def __init__(self, *a, **k):
        self.shape = (0,)
        self.dtype = None
        self.device = "cpu"
    def to(self, *a, **k):
        return self
    def numpy(self):
        import numpy as np
        return np.zeros(0)
    def item(self):
        return 0.0
    def __getitem__(self, k):
        return self


_install_torch_stub()


@pytest.fixture(autouse=True)
def setup_test_env():
    """Add project root to sys.path so local modules import correctly."""
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    yield
