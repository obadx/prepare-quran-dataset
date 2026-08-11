"""Shared pytest fixtures."""

from types import SimpleNamespace

import pytest


@pytest.fixture
def mask_stub():
    """Minimal stand-in for a ``MuaalemConformerEncoder`` in mask tests.

    ``_create_masks`` is effectively a pure function of its arguments plus two
    encoder attributes, so the golden mask tests can call it unbound against
    this stub instead of constructing a real encoder (which would pull in NeMo
    and allocate a 17-layer model just to build a boolean matrix).
    """
    return SimpleNamespace(
        self_attention_model="rel_pos",
        att_context_style="chunked_limited",
    )
