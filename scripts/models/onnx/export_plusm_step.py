#!/usr/bin/env python3
"""Export plus-M's MBart decode as a clean STATIC-KV single-step ONNX (no Loop,
no nested If) for a C++ continuous-batching host loop, plus the cross-KV prep
graph. Mirrors the PP-FormulaNet-S fastdec split, scaled to plus-M
(512-dim, 6-layer, head_dim 32, SINGLE-token greedy -- no MTP).

Weight map (deterministic, lifted from the fused inference_trt.onnx):
  lm = linear_231 ; memory_projector = linear_292 (2048->512)
  layer l: linears 232+10l + [self.k,self.v,self.q,self.out, cross.k,cross.v,cross.q,cross.out, fc1,fc2]
  LN: embed = create_parameter_14/15 ; layer l [self,cross,final] = layer_norm_{73+3l,74+3l,75+3l} ;
      decoder final = layer_norm_91

I/O (C++ binds):
  prep.onnx:  memory[B,144,2048] -> ck[6,B,16,144,32], cv[6,B,16,144,32]
  step.onnx:  tokens[B,1] int64, pos[1] int64, kb/vb[6,B,16,MAX,32], ck/cv[6,B,16,144,32]
              -> logits[B,1,50000], kb_out/vb_out[6,B,16,MAX,32]

Validated bit-exact vs the fused graph on the 12 golden crops before export.
"""
import sys, os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, onnxruntime as ort

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if len(sys.argv) < 2:
    sys.exit("usage: export_plusm_step.py <decoder_weights.npz> [MAXLEN=1056]  "
             "(bit-exact golden greedy JSON via PLUSM_GOLDEN env)")
WNPZ = sys.argv[1]
OUTDIR = f"{REPO}/models/formula/ppformulanet_plus_m"
MAX = int(sys.argv[2]) if len(sys.argv) > 2 else 1056
D, H, Dh, L, VOCAB = 512, 16, 32, 6, 50000
SC, ES = Dh ** -0.5, float(np.sqrt(D))
DEV = "cpu"
W = np.load(WNPZ)
def t(n): return torch.tensor(W[n.replace('.', '__')], dtype=torch.float32, device=DEV)


class Prep(nn.Module):
    """memory[B,144,2048] -> ck,cv[6,B,16,144,32] (cross-attn K/V, computed once)."""
    def __init__(s):
        super().__init__()
        s.register_buffer("mpw", t("linear_292.w_0")); s.register_buffer("mpb", t("linear_292.b_0"))
        for l in range(L):
            b = 232 + 10 * l
            s.register_buffer(f"ckw{l}", t(f"linear_{b+4}.w_0")); s.register_buffer(f"ckb{l}", t(f"linear_{b+4}.b_0"))
            s.register_buffer(f"cvw{l}", t(f"linear_{b+5}.w_0")); s.register_buffer(f"cvb{l}", t(f"linear_{b+5}.b_0"))

    def _shp(s, x):
        B, T, _ = x.shape; return x.view(B, T, H, Dh).permute(0, 2, 1, 3)

    def forward(s, memory):
        mp = memory @ s.mpw + s.mpb
        ck = torch.stack([s._shp(mp @ getattr(s, f"ckw{l}") + getattr(s, f"ckb{l}")) for l in range(L)])
        cv = torch.stack([s._shp(mp @ getattr(s, f"cvw{l}") + getattr(s, f"cvb{l}")) for l in range(L)])
        return ck, cv


class Step(nn.Module):
    """One static-KV decode step (single token)."""
    def __init__(s):
        super().__init__()
        s.register_buffer("emb", t("embedding_3.w_0")); s.register_buffer("posemb", t("m_bart_learned_positional_embedding_3.w_0"))
        s.register_buffer("lm", t("linear_231.w_0"))
        s.register_buffer("elnw", t("create_parameter_14.w_0")); s.register_buffer("elnb", t("create_parameter_15.w_0"))
        s.register_buffer("dflnw", t("layer_norm_91.w_0")); s.register_buffer("dflnb", t("layer_norm_91.b_0"))
        for l in range(L):
            b = 232 + 10 * l
            for off, nm in enumerate(['sk', 'sv', 'sq', 'so', 'ck', 'cv', 'cq', 'co', 'f1', 'f2']):
                s.register_buffer(f"{nm}w{l}", t(f"linear_{b+off}.w_0")); s.register_buffer(f"{nm}b{l}", t(f"linear_{b+off}.b_0"))
            s.register_buffer(f"slnw{l}", t(f"layer_norm_{73+3*l}.w_0")); s.register_buffer(f"slnb{l}", t(f"layer_norm_{73+3*l}.b_0"))
            s.register_buffer(f"clnw{l}", t(f"layer_norm_{74+3*l}.w_0")); s.register_buffer(f"clnb{l}", t(f"layer_norm_{74+3*l}.b_0"))
            s.register_buffer(f"flnw{l}", t(f"layer_norm_{75+3*l}.w_0")); s.register_buffer(f"flnb{l}", t(f"layer_norm_{75+3*l}.b_0"))

    def _ln(s, x, w, b): return F.layer_norm(x, (D,), w, b, 1e-5)
    def _shp(s, x):
        B, T, _ = x.shape; return x.view(B, T, H, Dh).permute(0, 2, 1, 3)
    def _mrg(s, x):
        B, _, T, _ = x.shape; return x.permute(0, 2, 1, 3).reshape(B, T, D)

    def forward(s, tokens, pos, kb, vb, ck, cv):
        # tokens[B,1] int64, pos[B] int64 (PER-SEQUENCE position -> continuous batching).
        B = tokens.shape[0]
        x = s.emb[tokens.reshape(-1)].view(B, 1, D) * ES
        pidx = pos + 2  # [B]
        x = s._ln(x + s.posemb[pidx].view(B, 1, D), s.elnw, s.elnb)
        col = torch.arange(MAX, device=tokens.device)
        mask = (col.view(1, 1, 1, MAX) > pos.view(B, 1, 1, 1)).to(x.dtype) * (-65504.0)  # [B,1,1,MAX]
        widx = pos.view(B, 1, 1, 1).expand(B, H, 1, Dh)  # per-row KV-write index (dim 2)
        nkb, nvb = [], []
        for l in range(L):
            r = x; h = s._ln(x, getattr(s, f"slnw{l}"), getattr(s, f"slnb{l}"))
            q = s._shp(h @ getattr(s, f"sqw{l}") + getattr(s, f"sqb{l}"))
            k = s._shp(h @ getattr(s, f"skw{l}") + getattr(s, f"skb{l}"))
            v = s._shp(h @ getattr(s, f"svw{l}") + getattr(s, f"svb{l}"))
            kbl = kb[l].scatter(2, widx, k); vbl = vb[l].scatter(2, widx, v)
            nkb.append(kbl); nvb.append(vbl)
            a = F.softmax((q @ kbl.transpose(-1, -2)) * SC + mask, -1) @ vbl
            x = r + (s._mrg(a) @ getattr(s, f"sow{l}") + getattr(s, f"sob{l}"))
            r = x; h = s._ln(x, getattr(s, f"clnw{l}"), getattr(s, f"clnb{l}"))
            q = s._shp(h @ getattr(s, f"cqw{l}") + getattr(s, f"cqb{l}"))
            a = F.softmax((q @ ck[l].transpose(-1, -2)) * SC, -1) @ cv[l]
            x = r + (s._mrg(a) @ getattr(s, f"cow{l}") + getattr(s, f"cob{l}"))
            r = x; h = s._ln(x, getattr(s, f"flnw{l}"), getattr(s, f"flnb{l}"))
            x = r + (F.gelu(h @ getattr(s, f"f1w{l}") + getattr(s, f"f1b{l}")) @ getattr(s, f"f2w{l}") + getattr(s, f"f2b{l}"))
        x = s._ln(x, s.dflnw, s.dflnb)
        logits = x @ s.lm                      # [B,1,VOCAB]
        next_token = logits.argmax(-1)         # [B,1] int64 — in-graph greedy argmax
        return logits, next_token, torch.stack(nkb), torch.stack(nvb)


@torch.no_grad()
def host_decode(prep, step, memory, start_id=2, max_len=400):
    B = memory.shape[0]
    ck, cv = prep(memory)
    kb = torch.zeros(L, B, H, MAX, Dh); vb = torch.zeros(L, B, H, MAX, Dh)
    ids = torch.full((B, 1), start_id, dtype=torch.long)
    out = [[] for _ in range(B)]; done = [False] * B
    for stp in range(max_len):
        pos = torch.full((B,), stp, dtype=torch.long)  # lockstep validation (all rows same pos)
        lg, ntg, kb, vb = step(ids, pos, kb, vb, ck, cv)
        nt = lg[:, -1, :].argmax(-1)
        assert torch.equal(ntg.view(-1), nt), "in-graph argmax != host argmax"  # gate: next_token matches
        for b in range(B):
            if done[b]: continue
            tk = int(nt[b])
            if tk == 2: done[b] = True
            elif tk not in (0, 1): out[b].append(tk)
        ids = nt.view(B, 1)
        if all(done): break
    return out


def main():
    prep = Prep().eval(); step = Step().eval()
    crops = np.fromfile(f"{REPO}/models/formula/ppformulanet_s/fast/golden3_crops.bin", dtype=np.float32).reshape(-1, 1, 384, 384)
    so = ort.SessionOptions(); so.log_severity_level = 3
    enc = ort.InferenceSession(f"{OUTDIR}/encoder.onnx", so, providers=["CPUExecutionProvider"])
    mem = torch.tensor(np.concatenate([enc.run(None, {"x": crops[i:i+1]})[0] for i in range(crops.shape[0])], 0))
    golden = os.environ.get("PLUSM_GOLDEN")
    if not golden or not os.path.exists(golden):
        sys.exit("set PLUSM_GOLDEN to the bit-exact golden greedy JSON before exporting")
    fused = json.load(open(golden))
    SUF = "" if MAX == 1056 else f"_{MAX}"  # bucket exports get a suffix; 1056 keeps prod names
    outs = host_decode(prep, step, mem, max_len=min(400, MAX))
    ok = sum(outs[i] == fused[str(i)] for i in range(len(outs)))
    print(f"[torch static-KV] {ok}/{len(outs)} bit-exact vs fused (MAX={MAX})")
    if ok != len(outs):
        print("FAILED -- not exporting"); sys.exit(1)

    B0 = 4
    if not SUF:  # prep is MAX-independent (cross-KV, CTX=144) — export only for the 1056 base
        dm = (torch.zeros(B0, 144, 2048),)
        torch.onnx.export(prep, dm, f"{OUTDIR}/prep.onnx", input_names=["memory"], output_names=["ck", "cv"],
                          opset_version=17, dynamic_axes={"memory": {0: "B"}, "ck": {1: "B"}, "cv": {1: "B"}}, dynamo=False)
    ds = (torch.zeros(B0, 1, dtype=torch.long), torch.zeros(B0, dtype=torch.long),
          torch.zeros(L, B0, H, MAX, Dh), torch.zeros(L, B0, H, MAX, Dh),
          torch.zeros(L, B0, H, 144, Dh), torch.zeros(L, B0, H, 144, Dh))
    torch.onnx.export(step, ds, f"{OUTDIR}/decoder_step{SUF}.onnx",
                      input_names=["tokens", "pos", "kb", "vb", "ck", "cv"],
                      output_names=["logits", "next_token", "kb_out", "vb_out"], opset_version=17,
                      dynamic_axes={"tokens": {0: "B"}, "pos": {0: "B"}, "kb": {1: "B"}, "vb": {1: "B"}, "ck": {1: "B"}, "cv": {1: "B"},
                                    "logits": {0: "B"}, "next_token": {0: "B"}, "kb_out": {1: "B"}, "vb_out": {1: "B"}}, dynamo=False)
    print(f"exported decoder_step{SUF}.onnx (MAX={MAX}) to {OUTDIR}")


if __name__ == "__main__":
    main()
