"""Shared pytest configuration.

Forces the test suite to run offline and on CPU only (OPS-03). The variables are
set at import time so they take effect before any test module imports
``transformers`` or ``torch``.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
