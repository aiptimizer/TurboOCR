#!/usr/bin/env python3
"""Raw-socket client simulating turboocr's C++ side: send decoded BGR crops
(no PNG/base64/JSON/HTTP) to the in-process vLLM sidecar; validate + time."""
import socket, struct, os, json, time, sys, threading, itertools
import concurrent.futures as cf
import cv2
SOCK = "/tmp/vl_sidecar.sock"
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "omnidocbench", "data", "images")
d = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "omnidocbench", "data", "OmniDocBench.json")))
forms, tbls = [], []
for page in d:
    p = os.path.join(IMG, page.get("page_info", {}).get("image_path", ""))
    if not os.path.exists(p): continue
    for a in page.get("layout_dets", []):
        ct=a.get("category_type","");poly=a.get("poly",[0,0,0,0]);xs=poly[0::2];ys=poly[1::2]
        b=[int(min(xs)),int(min(ys)),int(max(xs)),int(max(ys))]
        if b[2]-b[0]<8 or b[3]-b[1]<8: continue
        if ct=="equation_isolated" and a.get("latex","").strip() and len(forms)<30: forms.append((cv2.imread(p)[b[1]:b[3],b[0]:b[2]],0))
        elif ct=="table" and a.get("html","").strip() and len(tbls)<15: tbls.append((cv2.imread(p)[b[1]:b[3],b[0]:b[2]],1))
    if len(forms)>=30 and len(tbls)>=15: break
items=[(c.copy(),t) for c,t in (forms+tbls) if c is not None and c.size]
print(f"[client] {len(forms)} formula + {len(tbls)} table crops",flush=True)
_ids=itertools.count(1); lock=threading.Lock()

def call(crop, task):
    H,W=crop.shape[:2]
    rid=next(_ids)
    body=struct.pack("<IBII",rid,task,H,W)+crop.astype('uint8').tobytes()
    s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect(SOCK); s.sendall(body)
    def rn(n):
        bb=b""
        while len(bb)<n:
            c=s.recv(n-len(bb))
            if not c: raise IOError("closed")
            bb+=c
        return bb
    rid2,st,ln=struct.unpack("<III",rn(12)); txt=rn(ln).decode("utf-8","replace"); s.close()
    return st,txt

# correctness sample
print("\n=== sample outputs (no HTTP, raw BGR over socket) ===",flush=True)
for crop,task in (forms[:2]+tbls[:1]):
    st,txt=call(crop,task)
    kind="FORMULA" if task==0 else "TABLE"
    print(f"  {kind} status={st}: {txt[:90].strip()}",flush=True)

# throughput
pool=items*3
for C in (64,128,256):
    t0=time.time()
    with cf.ThreadPoolExecutor(max_workers=C) as ex: list(ex.map(lambda it:call(it[0],it[1]),pool))
    dt=time.time()-t0
    print(f"  c={C:3d}: {len(pool)/dt:6.1f} items/s   ({len(pool)} in {dt:.1f}s)",flush=True)
