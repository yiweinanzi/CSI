import json
import sys

sys.path.insert(0, '/root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/code/CSI-PAIRS-v2.0-server')
from formal_v2.formal_cli import _acquire_legacy_read_lock


try:
    lock = _acquire_legacy_read_lock('/root/xunlian/Futaoran/CSI_STREAMING_GPU_FIX_20260908/runs/complete-D-2ce2472')
except RuntimeError as error:
    assert isinstance(error.__cause__, BlockingIOError), repr(error)
    print(json.dumps({'passed': True, 'check': 'D active writer excludes a competing read lock', 'observed_cause': type(error.__cause__).__name__, 'D_interrupted': False}))
else:
    lock.release()
    raise AssertionError('Expected the running D writer to hold its output guard')
