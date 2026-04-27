"""
DeepBach/metadata.py — legacy stub (not used by the current codebase)

All metadata classes used by DeepBach are defined in:
    DatasetManager/metadata.py  (FermataMetadata, TickMetadata, KeyMetadata, …)

This file existed in the original DeepBach repo but is never imported by
model_manager.py, voice_model.py, or __init__.py.  It is kept here as an
empty stub so that any third-party code that does `import DeepBach.metadata`
does not crash with a ModuleNotFoundError.

The original version had two fatal bugs that made it un-importable on
Python 3.9 + PyTorch 2.x:
  1. `from .data_utils import SUBDIVISION` — SUBDIVISION was never defined in
     data_utils.py, causing an ImportError at module load time.
  2. `raise NotImplementedError` inside Metadata.__init__ — prevented every
     subclass from calling super().__init__() without raising immediately.

If you need custom metadata, add it to DatasetManager/metadata.py instead.
"""
