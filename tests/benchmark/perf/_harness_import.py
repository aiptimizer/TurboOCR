"""Import shim so bench scripts can `from _harness_import import _harness_mod as H`.

pytest and standalone scripts need to find _harness.py regardless of cwd.
"""

import importlib.util
import sys
from pathlib import Path

_MOD_NAME = "_turbo_ocr_bench_harness"
if _MOD_NAME in sys.modules:
    _harness_mod = sys.modules[_MOD_NAME]
else:
    _path = Path(__file__).resolve().parent / "_harness.py"
    _spec = importlib.util.spec_from_file_location(_MOD_NAME, _path)
    _harness_mod = importlib.util.module_from_spec(_spec)
    sys.modules[_MOD_NAME] = _harness_mod
    assert _spec.loader is not None
    _spec.loader.exec_module(_harness_mod)
