"""Test-only: the pre-nnterp names that the A/B tests' old code reads off the model.

The A/B tests (``test/test_*_port.py``) load pre-port modules from git and run them on the nnterp
model. That code reads ``mt._model`` (the HF module), ``mt.layer_name_format`` and
``mt.attn_module_name_format`` (native module paths, the layout of the reported families), which
``ModelandTokenizer`` no longer has. The tests build their model from this subclass; nothing under
``src/`` or ``scripts/`` uses it.
"""

import torch

from src.models import ModelandTokenizer


class LegacyModelandTokenizer(ModelandTokenizer):
    """``ModelandTokenizer`` plus the three names the pre-port code reads."""

    layer_name_format = "model.layers.{}"
    attn_module_name_format = "model.layers.{}.self_attn"

    @property
    def _model(self) -> torch.nn.Module:
        """The HF module (``self._module``)."""
        return self._module
