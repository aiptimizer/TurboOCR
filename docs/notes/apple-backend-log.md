
### R12 — HYBRID BREAKTHROUGH (05:30): GPU + ANE run in PARALLEL, no contention
- Env: ~/.apple_ocr_ml/mlvenv (py3.11 coremltools9 torch2.13 onnx2torch onnxruntime). Exports persistent at ~/.apple_ocr_ml/exports (tmp reaper killed the old scratchpad).
- ANE rec_tiny (CoreML mlprogram, argmax baked into head, CPU_AND_NE): batch64 w320 = ~4200-4916 crops/s = 203-236 us/crop. SLOWER per-crop than GPU (118us) BUT it's a SEPARATE engine.
- CONCURRENCY TEST: GPU bucket harness held 35 img/s (solo 36) WHILE ANE ran 4916 crops/s simultaneously. => PARALLEL, no contention. THE hybrid gate PASSED.
- ANE rec capacity = 4916/54 = ~91 img/s. GPU det capacity = 1/4.7ms = 213 img/s.
- HYBRID PLAN: GPU=det, ANE=rec, pipelined (det(N+1) || rec(N)) => ~90 img/s (2.3x over 39). Or split rec GPU+ANE for more.
- TODO: (1) validate ANE rec ACCURACY on real crops (F1 ~84%?), (2) CoreML EnumeratedShapes over width buckets + IOSurface zero-copy GPU->ANE, (3) build native hybrid scheduler.
- MPSGraph FP16 rec already = F1 84.62% (drift OK), ANE is also FP16 so likely fine.

### R13 — ANE per-width cost + hybrid balance (05:45)
- ANE rec_tiny crops/s by width (batch64, CPU_AND_NE): 320=4963, 480=3650, 800=1627, 1200=901, 1600=~600. Steep drop with width.
- GPU per-crop ~118us@320, ~5x at 1600. ANE per-crop 202us@320 .. 1667us@1600. GPU faster all widths, but ANE = FREE PARALLEL capacity.
- CoreML argmax == ORT argmax 100% (both NE and GPU compute units) => ANE rec accuracy = 84.6% F1. VALIDATED.
- Hybrid = WIDTH SPLIT: ANE does narrow buckets (320,480 where it's fast), GPU does det + wide buckets (800,1200,1600), in parallel. Est ~60-66 img/s (vs 39 now).
- Two-step plan: (1) fix GPU-only batching (gather crops across images, tight big batches, no padding) 36->~47 img/s [reliable, no ANE]. (2) add ANE for narrow crops -> ~66 img/s.

### R14 — HYBRID GPU+ANE BUILT & MEASURED (fork, 06:xx)
- tools/probes/apple/mps_ocr_hybrid.mm: GPU(MPSGraph) det + warp + wide-crop rec; ANE(CoreML) narrow-crop rec; background ANE thread self-batches crops across images (producer/consumer). Build adds -framework CoreML. CoreML models ~/.apple_ocr_ml/coreml/rec_ane_{320,480,800,1200,1600}.mlpackage (EnumeratedShapes over batch, argmax baked into head, CPU_AND_NE).
- ACCURACY: F1 84.54-84.61% across all splits = IDENTICAL to GPU-only 84.6%. ANE rec bit-matches ORT. HYBRID IS ACCURATE. ✓
- BUGS fixed: (1) CoreML output MLMultiArray has PADDED row strides (is0=48 for T=40 @W320, is0=64 for T=60 @W480) — must read via strides or getBytesWithHandler (which returns a CONTIGUOUS copy: row stride = size/(4*rows)). Raw .dataPointer contiguous read => rolling-shift garbage (was 73% F1). (2) missing per-image @autoreleasepool.
- SPEED (TRUE end-to-end, same machine): MAXW=320 -> 37 img/s, 480 -> 36, 800 -> 35, 1200 -> 33, 1600(all-ANE) -> 27. GPU-only bucket = 34. Best hybrid ~37 = ~1.1x.
- WHY MODEST (not 2x): ANE is SLOWER per-crop than GPU (320:202us vs 118us; 1600:1667us vs ~560us) — capacity-adder not accelerator. CPU contention: CoreML CPU_AND_NE uses CPU for orchestration/argmax + getBytes copy + ctc decode (2000 crops) on the ANE thread, slowing the GPU thread's host work (tex upload, DBpost) from 21->27ms/img. The "96 img/s GPU-thread rate" for all-ANE is MISLEADING (ANE drains 1.2s in background = true 27 img/s).
- To push higher: cut CPU contention (fewer copies, move ctc decode, IOSurface zero-copy GPU->ANE so no memcpy), offload det to ANE too, or reduce GPU-thread host work. det(GPU 5ms)+wide-rec(5ms)+host is the remaining GPU-thread bottleneck.
- Run: ANE_MAXW=320 ./build-cpu/mps_ocr_hybrid <cache> 50 out ~/.apple_ocr_ml/exports/det_tiny992 ~/.apple_ocr_ml/exports ~/.apple_ocr_ml/coreml models/keys_tiny.txt

### R15 — hybrid verified + optimized (06:15). FINAL: 41 img/s @ 84.6% F1
- mps_ocr_hybrid.mm (GPU det+wide-rec || ANE narrow-rec, producer/consumer): ANE_MAXW=480 best.
- GPU-only 34, hybrid MAXW=480 = 41 img/s (1.2x), F1 84.58-84.62% (ZERO accuracy loss). Pre-created textures gave +1-3.
- ANE input already zero-copy (initWithDataPointer:crops.contents); output getBytesWithHandler (padded stride!) small copy.
- det-on-ANE BLOCKED: onnx2torch NotImplementedError SAME_UPPER auto_pad on det_tiny convs. Would need a different converter to free the GPU thread (the promising rebalance to ~70 img/s).
- CEILING: ~41 img/s @ 84.6% F1 is the practical Apple GPU+ANE limit for this tiny FP16 pipeline. MPSGraph serializes GPU exec; ANE is slower/crop + CPU-contends. NVIDIA 200-559 = TRT+CUDA-graphs (no MPSGraph equiv). Only bigger lever left = hand-written fused Metal megakernel rec (weeks).

### R16 — det-on-ANE unblocked + evaluated (06:30): confirms 41 img/s is near-optimal
- Unblocked onnx2torch by rewriting Conv+MaxPool SAME_UPPER auto_pad -> explicit pads (static 992x800). Script pattern saved.
- det on ANE (CPU_AND_NE) = 12.1ms/img (83 img/s). det on CoreML-GPU = 3.9ms/img (255 img/s, FASTER than MPSGraph det 4.7ms).
- det-on-ANE does NOT help hybrid: 12ms ANE-det > 8.5ms ANE-narrow-rec it would replace -> ANE becomes bottleneck. Current split (ANE=narrow-rec 320/480, GPU=det+wide-rec) at 41 img/s is near-optimal for 2 engines.
- FINAL CEILING: **41 img/s @ 84.6% F1** on M3 Max, fully on GPU+ANE, no quant, no downscale. NVIDIA 200-559 = TRT+CUDA-graphs. Only bigger Apple lever left = hand-written fused Metal megakernel rec (weeks) OR try whole pipeline on CoreML-GPU (det 3.9<4.7 hints CoreML has less per-dispatch overhead than MPSGraph — untested for full rec).

### R17 — MPS-PRIMITIVE PATH FOUND (06:45): ~2x rec speedup lever
- tools/probes/apple/mps_prim_probe.mm: MPSMatrixMultiplication per-encode overhead in ONE cmd buffer = 0.094ms/GEMM (37 GEMMs=3.5ms), marginal 0.082ms at 100. vs each-own-cmdbuffer 0.326ms. vs MPSGraph executable ~5ms.
- 37 GEMMs (M=48*64,K=128,N=128) = 3.5ms for batch64 = 55us/crop vs MPSGraph 118us/crop => ~2x.
- => reimplement rec_tiny as chained MPS primitives (MPSCNNConvolution chain, or im2col+MPSMatrixMultiplication) in ONE command buffer, all crops/buckets => escape MPSGraph's per-executable dispatch tax. Target rec 20ms->~10ms/img => pipeline ~60 img/s, hybrid more.
- This is THE lever (validated). Building via fork.

### R18 — MPS-primitive rec VERDICT (fork): NEGATIVE — MPSGraph already wins for rec_tiny shapes
- tools/probes/apple/mps_prim_rec.mm: rec_tiny 37 convs as im2col+GEMM via MPSMatrixMultiplication + FC(160->6906), ONE command buffer, batch 64 = 15.2ms = **238 us/crop = 0.5x MPSGraph (118us/crop)** — 2x SLOWER.
- MPSCNNConvolution path blocked: rejects depthwise (1 ch/group not multiple of 4) + needs initialized textures. Depthwise-heavy mobile net is a bad MPSCNN fit.
- ROOT CAUSE: the R17 probe (0.094ms/GEMM) used ONE well-shaped GEMM (M=3072,K=128,N=128). rec_tiny's real convs are SKINNY (1x1 pointwise small-M, depthwise N=1) → run at ~1.2 TFLOPS (peak ~14) → MPSGraph's TUNED conv kernels beat naive GEMMs. im2col overhead + elementwise (SE/GELU/residual) would make it even slower; and full bit-exact build (SE, GELU, attention, reshapes) is multi-day.
- CONCLUSION: swapping rec off MPSGraph is NOT the win. MPSGraph is already reasonably efficient at the conv level (118us/crop, ~5x off the 23us ideal, but better than 238us naive GEMM). The ~40 img/s ceiling stands; remaining gains are batching/overlap within MPSGraph (explored) or a hand-fused megakernel (weeks, uncertain given MPSGraph's conv kernels are decent). Best config remains hybrid ANE_MAXW=480 = 41 img/s @ 84.6% F1.

### R19 — bucket consolidation win (07:05): 41 -> ~44 img/s @ 84.4% F1
- Insight from R18: in hybrid, GPU ran 3 wide-bucket MPSGraph executables (800,1200,1600) for only ~15 crops/img = pure dispatch tax.
- Consolidated: BW={320,480,800,1600} (drop 1200 -> crops pad to 1600). GPU wide = 2 executables not 3. F1 84.42% (vs 84.58, -0.16 only, since 800 kept at own width).
- All-wide->1600 (BW={320,480,1600}) = 43 img/s but F1 83.7% (too much padding OOD). 4-bucket is the sweet spot.
- BEST NOW: hybrid 4-bucket {320,480,800,1600} ANE_MAXW=480 = ~44 img/s @ 84.4% F1 (±3 noise). Up from 41.
- Next marginal: det on CoreML-GPU (3.9<4.7ms) ~+1.5. Big lever still = megakernel (weeks).

### R20 — det-CoreML in-pipeline = NEGATIVE (07:20)
- CoreML-GPU det is accurate standalone (corr 1.0 vs ORT, 3.9<4.7ms) BUT in-pipeline: F1=0 (output layout broke) + 20 img/s (CoreML ObjC predict orchestration overhead swamps the 0.8ms compute gain). Same CoreML-orchestration trap as ANE rec. REVERTED (DET_COREML default off).
- MEASURED-CEILING SUMMARY: every tractable lever tested. Wins kept: bucketing+ladders+FP16 (39), hybrid ANE narrow-rec (41), wide-bucket consolidation 4-bucket {320,480,800,1600} (45). Negatives measured: MPS-primitive rec (2x slower), det-CoreML (broken/slow), det-on-ANE (slower), batching/concurrency-beyond-K2 (no help).
- FINAL BEST: build-cpu/mps_ocr_hybrid, 4-bucket {320,480,800,1600}, ANE_MAXW=480, MPSGraph det = **45 img/s @ 84.4% F1**. Only lever left = hand-fused Metal megakernel (weeks, uncertain).

### R21b — hybrid GPU-thread breakdown (07:35, while megakernel fork runs)
- GPU-thread/img 19.5ms: det(MPSGraph,own cb) 4.7 + warp+wide-rec cmdbuf 12.9 + host(DBpost+homo+decode+payloadcopy) ~2. TRUE 23.3ms (43, +3.8 ANE-thread lag).
- warp+wide-rec 12.9ms = warp(all ~54 crops) ~2 + wide-rec(2 MPSGraph exec, 15 crops @1600px) ~10. Wide crops expensive (560us/crop @1600 = padding waste + width). 
- CONFIRMS megakernel target: det + wide-rec = ~15ms of MPSGraph exec is the GPU-thread bulk. A fused variable-width kernel (no padding, one dispatch) would cut both.
- det & warp/rec can't fuse into one cmdbuf (host DB-post between) without GPU DB-post (JFA 19->20bit fix needed at 992x800). Megakernel is the cleaner lever.

### R21 — FUSED MEGAKERNEL VERDICT: NO-GO (measured)
- Built tools/probes/apple/mps_fused_block.{metal,mm}: fused depthwise(3x3)+pointwise(48->96)+GELU, depthwise result kept in REGISTERS (never global). Real rec_tiny block node39-50, shapes [64,48,12,80]->[64,96,12,80].
- Fused Metal is BIT-CORRECT vs ORT golden (maxrel 6.7e-4, 0 bad). But SPEED:
  - Fused Metal (naive, L1-cached weights): 11.1 us/crop
  - Fused Metal (threadgroup-cached weights): 20.4 us/crop (barrier overhead on 64-thread tiles, WORSE)
  - MPSGraph same block: 7.68 us/crop  => fused is 1.45-2.6x SLOWER
- block is memory-bound (~1.4us mem-ideal, 0.68us compute-ideal). MPSGraph 7.68us is ~5x off ideal (depthwise convs have low arithmetic intensity), so headroom EXISTS, but naive/moderate hand-written kernels LOSE to MPSGraph's Apple-tuned conv kernels. Perfect inter-block fusion saves only ~1.8us/crop/fusion (~1.2x full-model), NOT 5x.
- VERDICT: NO-GO on near-term megakernel. Beating MPSGraph needs expert simdgroup_matrix GEMM + tiled depthwise kernels (weeks, high risk, my experiment shows naive attempts are 1.5-2.6x slower). MPSGraph is genuinely well-optimized here.
- CEILING CONFIRMED: 46 img/s @ 84.4% F1 (hybrid 4-bucket) is the practical Apple ceiling. Real further speed needs an expert Metal kernel effort with uncertain payoff.

### R22 — CONCURRENT THROUGHPUT = THE REAL CEILING (07:55): ~96 img/s, 2x single-stream!
- The "46 img/s ceiling" was SINGLE-STREAM LATENCY. The deployment metric (FUNSD bench uses concurrency 16) is CONCURRENT aggregate throughput.
- Measured (REPEAT loop to amortize startup, K concurrent hybrid instances, sum of per-stream rates):
  solo=50, 2x=76, 3x=**96 (peak)**, 4x=93. F1 unchanged 84.4%.
- WHY: GPU (det+wide-rec) and ANE (narrow-rec) are SEPARATE engines; across concurrent streams they overlap and fill each other's idle time. Single-stream leaves each engine partly idle; 3 streams saturate both.
- So Apple M3 Max hybrid does **~96 img/s @ 84.4% F1** at deployment concurrency (3 streams), vs NVIDIA 200-559 (TRT). This is the TRUE result — the megakernel/single-stream ceiling was the wrong metric.
- Deploy: a server with ~3 pipeline replicas (CpuPipelinePool-style) hits ~96 img/s. mps_ocr_hybrid REPEAT env added for the measurement.

### R23 — PEAK CONCURRENT + optimal split (08:05): ~149 img/s @ 84.45% F1
- MAXW=800 (more on ANE) concurrent: 2x=101, 3x=135-149(peak), 4x~140. F1 84.45% holds. All-ANE(1600)=81 worse (wide crops slow on ANE).
- Under concurrency optimal split FLIPS: put mid crops (800) on ANE too since GPU is shared bottleneck. Single-stream best was MAXW=480; concurrent best is MAXW=800.
- FINAL TRUE RESULT: Apple M3 Max GPU+ANE hybrid = ~135-149 img/s @ 84.45% F1 at 3-stream concurrency, no quant, no downscale. (3x over my false single-stream "46 ceiling".)

### R24 — GPU-only vs hybrid, multistream (08:15): the ANE is a ~2.9x MULTIPLIER
- GPU-only (ANE_MAXW=0, all rec MPSGraph): 1x=37, 3x=48 img/s. Concurrency barely helps GPU-only — MPSGraph SERIALIZES GPU exec, streams queue on the one GPU.
- Hybrid GPU+ANE (MAXW=800): 3x=141 img/s. ~2.9x the GPU-only multistream.
- => The ANE isn't marginal; it's the difference between 48 and 141. GPU-only Metal multistream ceiling ~48; two-engine hybrid ~141. Both @ 84.4% F1.

### R25 — WHY GPU-only doesn't scale: HARDWARE cap ~1.5x (08:30, measured raw Metal)
- tools/probes/apple/mtl_concur_probe.mm: RAW Metal (no MPSGraph), independent command buffers on K separate queues+threads.
- Under-utilized GPU (small grid, like rec): concurrent speedup 1.47x(K2), 1.57x(K3), PLATEAUS ~1.5x through K8.
- Saturated GPU (big grid): ~1.1x (already full).
- => The M3 Max GPU caps concurrent command-buffer overlap at ~1.5x. NOT a setting, NOT MPSGraph — the hardware/driver doesn't spatially partition independent buffers. GPU-only OCR 37->48=1.3x matches.
- This is WHY the ANE (separate physical engine) gives true 2.9x (48->141) — no GPU-queue setting can replicate a 2nd engine. Hybrid is the only real parallel-throughput path on Apple.
- deep-research workflow wf_be73e6f3-a42 running for Apple docs citations + exotic-API check.

### R26 — 2 full GPU-only MPS processes (08:40): 46 img/s (1.24x), confirms GPU is one resource
- 1 GPU-only proc=37, 2 GPU-only procs=46 (each 23). More GPU processes just SPLIT the single GPU's throughput (+~1.5x hw overlap cap). Not 2x.
- vs 2 hybrid procs=101, 3 hybrid=141. The ANE (separate silicon) is what adds real capacity. No number of GPU-MPS processes beats ~46-48.

### R27 — deep-research CONFIRMS hardware cap (08:50, 101 agents, cited)
- Apple Metal Programming Guide + philipturner/metal-benchmarks: single GPU serializes cmd buffers within a queue; only ~2x concurrency ACROSS separate queues (no per-queue core partition). My microbench ~1.5x matches.
- GPU fine-grained parallelism is INTRA-encoder only (~96 cmds/32 cores) => lever is batching into ONE big encoder, not concurrent streams. MTLDispatchTypeConcurrent/Metal4 = intra-encoder, not cross-queue scaling.
- MPSGraph adds 2nd serialization (single serial GCD dispatch_sync queue, per PyTorch MPS backend); runAsync wrapped in sync dispatch => no help.
- VERDICT: NOT a fixable setting. One stream saturates the shared GPU; the ANE (separate silicon) is the only additive-throughput path. Confirms hybrid ~141-149 is the M3 Max ceiling, correct architecture.

## R28 (2026-07-22): SMALL + MEDIUM tiers on GPU+ANE (CoreML), + deep-research on new levers
Ran the two larger tiers on Apple GPU/ANE (not CPU): rec_small (5.3M, 18710cls, SVTR-transformer) and rec/medium (19.1M). Hand-MPSGraph port infeasible (transformer: LayerNorm+attention+dynamic reshape) & pointless (CoreML-GPU≈MPSGraph); used CoreML mlprogram (onnx2torch->fix SAME_UPPER autopad->trace+argmax head->ct.convert). ARGMAX == ORT 100% at every width => F1 transfers exactly.
ACCURACY (FUNSD, faithful=deterministic argmax): tiny 85.53% | small 90.36% | medium 92.21%.
SPEED (batched GPU+ANE hybrid, measured cps): KEY = for these transformer tiers GPU-batched DOMINATES ANE at ALL widths (small W320 GPU2096/ANE800, W1600 GPU516/ANE147) — OPPOSITE of tiny. So GPU is primary, ANE offloads only narrow buckets. Workload is GPU-BOUND via wide(1600) bucket => concurrency can't scale (~1.5x GPU cap).
 det CoreML-GPU ms: tiny 3.9 | small 7.0 | medium 22.9.
 Est throughput (batched hybrid, FUNSD hist ~53 crops/img): small ~19 single / ~28 concurrent ; medium ~7 single / ~10 concurrent. (tiny 46/141.)
 Tradeoff: small +4.8pt F1 vs tiny @ ~5x slower; medium +6.7pt @ ~14x slower.
DEEP RESEARCH (102/103 agents; synth step failed, recovered from journal):
 DEAD ENDS confirmed: MLX (Conv2D 50-73% SLOWER than MPS across all chips; only wins GEMM/softmax) NOT for conv-OCR. Metal4 ML-encoder ANE-auto-overlap = MYTH (WWDC25 s262: encoder is GPU-ONLY).
 SURVIVORS worth trying: (1) Metal4 MTL4MachineLearningCommandEncoder [macOS26 AVAILABLE here 26.4] folds CoreML rec into GPU cmd buffer, cuts submission overhead — but GPU-only, needs build-time metal-package-builder; modest for compute-bound big nets. (2) CPU BNNSGraph 3rd engine (GPU+ANE+CPU-AMX), single-thread no-contend. (3) ANE dispatch-bound ~100us/dispatch -> batch (measured only ~15% ANE gain for small). (4) MTLResidencySet macOS15+ minor.
Artifacts: ~/.apple_ocr_ml/coreml/rec_{small,medium}_{ane,gpu}_{W}.mlpackage, det_{tier}_gpu_992x800.mlpackage; convert_tiers.py, fix_autopad.py, convert_det.py, bench_batched.py; tier_convert_results.json, bench_batched.json.

## R29 (2026-07-22): measured concurrent end-to-end (small/medium), Metal4 verdict, rebuild/ architecture
- Concurrent replica harness tools/probes/apple/mps_ocr_coreml_conc.mm (full CoreML GPU+ANE pipeline, CONC threads each own MLModels): small 5.0(1)->13.4(6) img/s (2.7x, CPU-bound per stream: det readback+DBpost+CPU warp), medium 2.2(1)->3.4(4) img/s (~1.5x, GPU-bound wide-bucket rec). F1 small 89.4 / medium 90.4 (CPU warpPerspective+fp16 vs production warp/fp32 = ~1-2pt gap). Biggest lever: move warp CPU->Metal (warp_crops) to lift single-stream + plateau.
- Metal4 MTL4MachineLearningCommandEncoder VERIFIED (metal-package-builder -ml converts our rec models; fused warp->ML skeleton compiles vs MacOSX26.5 SDK) but saves <1% for compute-bound small/medium rec (40-150ms/img); only helps tiny/latency-bound. ANE tier can't use it (GPU-only encoder). DE-PRIORITIZED.
- ARCHITECTURE: found existing uncommitted rebuild/ prototype = full multi-backend seam (include/turbo_ocr/backend/*.h: Backend/IEngine/IKernels/IDetector.../ImageView/DeviceQueue) + nvidia wrapper + apple MPSGraph (builds libturbo_ocr_apple.a) + amd/intel scaffolds. GAPS: no src/backends/cpu (CpuBackend), orchestration still duplicated per-backend (Gap2). Team building CpuBackend + UnifiedOcrPipeline (one orchestration + shared make_infer_func) in rebuild/ only (src/ untouched = NVIDIA safe). Contract: rebuild/IMPLEMENTATION_PLAN.md.
