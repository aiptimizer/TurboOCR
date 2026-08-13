#!/usr/bin/env python3
"""Function-length ratchet.

A long function is the defect this tree produced most reliably: five HTTP
handlers between 135 and 323 lines, each mixing request admission with the work
it admits, so which side of that line a given check sat on was invisible. They
were split by hand; this is what stops them growing back, and what stops the
next one arriving.

MAX is deliberately loose. It is a backstop against 300-line lambdas, not a
style opinion about 60 versus 90 — the aim is a gate nobody is tempted to
disable. `allowed` is a RATCHET: entries come off as functions shrink and none
are ever added to make a build pass (see task #16 for the queue).

Exit status is the gate: non-zero when a function outside the ratchet is over.
"""

import os
import re
import sys

MAX = 180
allowed = {
  # RATCHET — over the limit today, tracked in task #16. Same rule as the file
  # ratchet: entries come off as they shrink, none are ever added.
  "src/service/server/bootstrap/server_config.cpp",
  "src/service/http/pdf/stream_route.cpp",
  "src/service/http/pdf/pdf_request.cpp",
  "src/backends/intel/engine/openvino_engine.cpp",
  "src/service/http/unified_routes.cpp",
  "src/pdf/text/font_style.cpp",
}
bad=[]
seen=set()
for root,_,files in os.walk("src"):
    for f in files:
        if not f.endswith((".cpp",".cu")): continue
        p=os.path.join(root,f)
        seen.add(p)
        L=open(p,errors="ignore").read().split("\n")
        start=None
        for i,l in enumerate(L):
            if re.match(r'^[A-Za-z_][\w:<>,&\* ]*\s+[\w:]+\(.*\)\s*\{$|^[A-Za-z_][\w:<>,&\* ]*\s+[\w:]+\([^;]*$', l) and not l.startswith(("//","#")):
                start=i
            if l=="}" and start is not None:
                if i-start > MAX and p not in allowed:
                    bad.append(f"{p}:{start+1} ({i-start} lines)")
                start=None
# A ratchet entry that matches no file is a silently-dropped exemption — that is
# how a plain file MOVE turned this gate red for days (server_config.cpp moved
# into bootstrap/ and its entry stopped matching anything). Stale entries fail
# loudly so a rename updates the ratchet in the same commit.
stale = allowed - seen
for s in sorted(stale): print("  ratchet entry matches no file (moved or deleted?): " + s)
for b in bad: print("  "+b)
sys.exit(1 if bad or stale else 0)
