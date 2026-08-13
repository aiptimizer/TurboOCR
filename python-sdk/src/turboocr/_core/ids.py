from __future__ import annotations

import os
import time
import uuid as _uuid


def make_uuid7() -> str:
    # uuid.uuid7 added in Python 3.14 stdlib; fall back to manual construction
    # on older runtimes. getattr keeps both branches type-checkable.
    uuid7 = getattr(_uuid, "uuid7", None)
    if uuid7 is not None:
        return str(uuid7())
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF
    rand_b = int.from_bytes(os.urandom(8), "big") & 0x3FFFFFFFFFFFFFFF
    uuid_int = (ts_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    hex_str = f"{uuid_int:032x}"
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"


def short_request_id() -> str:
    return make_uuid7().replace("-", "")[:16]
