<p align="center">
  <sub>v4 alpha · 后端：NVIDIA、Apple（Metal + 神经引擎）、Intel（OpenVINO）、AMD（ROCm）、CPU · AMD 尚未在真实硬件上验证</sub>
</p>

<p align="center">
  <img src="tests/benchmark/comparison/images/hero_banner.svg" alt="TurboOCR — 最快的 GPU 文档解析器。单张 RTX 5090 上整页 OCR 650+ 张/秒。" width="100%">
</p>

<p align="center">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <strong>最快的 GPU 文档解析器 — OCR · 版面 · 表格 · 公式 → Markdown，单卡 650+ 张/秒。</strong><br>
  C++ / CUDA / TensorRT / PP-OCRv6 &mdash; Linux + NVIDIA GPU
</p>

<h3 align="center">v4.0-alpha — 一套流水线，多种后端</h3>
<p align="center">
  <sub>统一引擎 + 设备抽象层：NVIDIA（已发布）· Apple Metal + 神经引擎、Intel OpenVINO（测试中）· AMD ROCm（开发中）· 原生 Python 库 · PP-OCRv6 <code>tiny</code>/<code>small</code>/<code>medium</code> 三档 · <a href="docs/guides/upgrading-v4.md">v4 变更说明</a></sub>
</p>

<p align="center">
  <a href="https://github.com/aiptimizer/TurboOCR"><strong>⭐ 在 GitHub 上给 TurboOCR 点个 Star</strong></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/throughput-650%2B_img%2Fs-blue?style=flat-square&logo=speedtest&logoColor=white" alt="650+ 张/秒">
  <a href="https://turboocr.com"><img src="https://img.shields.io/badge/website-turboocr.com-3B82F6?style=flat-square&logo=googlechrome&logoColor=white" alt="turboocr.com"></a>
  <a href="https://github.com/aiptimizer/TurboOCR/releases/latest"><img src="https://img.shields.io/github/v/release/aiptimizer/TurboOCR?style=flat-square&logo=github&logoColor=white" alt="Release"></a>
  <a href="https://ghcr.io/aiptimizer/turboocr"><img src="https://img.shields.io/badge/docker-ghcr.io-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker"></a>
  <img src="https://img.shields.io/badge/C%2B%2B20-00599C?style=flat-square&logo=cplusplus&logoColor=white" alt="C++20">
  <img src="https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/TensorRT-10.15-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="TensorRT 10.15">
  <img src="https://img.shields.io/badge/Metal_%2B_ANE-000000?style=flat-square&logo=apple&logoColor=white" alt="Apple Metal + 神经引擎">
  <img src="https://img.shields.io/badge/OpenVINO-1E7BD9?style=flat-square" alt="Intel OpenVINO">
  <img src="https://img.shields.io/badge/ROCm-ED1C24?style=flat-square&logo=amd&logoColor=white" alt="AMD ROCm（开发中）">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 库">
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square&logo=opensourceinitiative&logoColor=white" alt="MIT License">
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> &middot;
  <a href="#基准测试">基准测试</a> &middot;
  <a href="#提升精度">精度</a> &middot;
  <a href="#模型">模型</a> &middot;
  <a href="#python">Python</a> &middot;
  <a href="#api">API</a> &middot;
  <a href="docs/index.md">文档</a>
</p>

---

TurboOCR 是一个完整的 GPU 文档解析器。它把 PP-OCRv6 文本检测与识别、
版面分析、表格识别（HTML）、公式识别（LaTeX）以及按阅读顺序导出的 Markdown
整合进一条设备驻留的流水线，通过 HTTP 和 gRPC 提供服务。所有推理都在本地完成，
不依赖视觉语言模型，也不调用任何外部 API。

在单张 RTX 5090 上实测：FUNSD 表单整页 OCR 超过 650 张/秒，完整结构化解析 20
页/秒；同类 VLM 文档解析器约为每秒 1 页（[基准测试](#基准测试)）。

- **模型。** 一个 PP-OCRv6 模型覆盖拉丁文、中文与日文，分 `tiny`、`small`、`medium` 三档（`tiny` 不含日文假名，日文请用 `small`/`medium`）；另有阿拉伯文、西里尔文、韩文、泰文、希腊文专用识别器。
- **文档结构。** PP-DocLayoutV3 版面分析、SLANet+ 表格转 HTML、PP-FormulaNet_plus-S 公式转 LaTeX、按类别感知的阅读顺序。每个阶段都按请求开启；默认路径不为未使用的阶段付出任何代价。
- **PDF。** 原生渲染与识别，支持自动转正、逐页流式输出、整本 PDF 导出 Markdown。
- **后端。** 同一条流水线运行在 NVIDIA CUDA/TensorRT、Apple Metal + 神经引擎、Intel OpenVINO、AMD ROCm 以及纯 CPU 上。
- **Python。** C++ 流水线以原生 Python 库形式提供，内置副本池（[Python](#python)）。
- **运维。** 一行 Docker 部署，Prometheus 指标，单个二进制同时提供 HTTP 与 gRPC。

完整文档：**[docs/](docs/index.md)**

---

## 快速开始

选择你的硬件。完整的分步安装指南（覆盖所有后端与 Python 库）见
**[安装 — 选择硬件](docs/getting-started/install.md)**。

<details open>
<summary><strong>NVIDIA GPU</strong> &nbsp;·&nbsp; 已发布</summary>

Linux、驱动 595+、Turing 或更新架构。纯文本约 4 GB 显存，完整流水线约 8 GB。

```bash
docker run --gpus all -p 8000:8000 -p 50051:50051 \
  -v trt-cache:/home/ocr/.cache/turbo-ocr \
  ghcr.io/aiptimizer/turboocr:latest
```

首次启动会构建 TensorRT 引擎（5090 约 90 秒，旧卡最长一小时；`TRT_OPT_LEVEL=3`
可缩短 3–5 倍）。命名卷会缓存引擎，之后启动即秒开。所有权重都内置在镜像里，
环境变量决定加载哪些：

```bash
-e TABLE_BACKEND=slanext              # 表格 → HTML
-e FORMULA_BACKEND=ppformulanet_s     # 公式 → LaTeX
-e OCR_MODEL=medium                   # tiny（默认）| small | medium | arabic | eslav | korean | thai | greek
```

→ [Docker 与部署](docs/getting-started/docker.md) · [原生构建](docs/getting-started/native.md)
</details>

<details>
<summary><strong>Apple Silicon</strong> &nbsp;·&nbsp; 测试中 &nbsp;·&nbsp; Metal + 神经引擎，仅原生运行</summary>

检测与透视变换在 GPU（Metal + MPSGraph）上执行；识别是 GPU + 神经引擎混合方案 —
窄行文本通过 CoreML 在 ANE 上与 GPU 并行推理。容器无法运行：macOS 虚拟化不暴露 GPU。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON
cmake --build build -j"$(sysctl -n hw.ncpu)"

# 固定 9 档识别宽度（否则自动发现会构建 42 档，吞吐大约减半）：
export TURBO_APPLE_REC_BUCKETS=320,480,800,1200,1600,2000,2500,3200,4000

./build/turboocr-server --backend apple
```

提供相应模型后，版面、表格、公式与自动转正同样可用。详见
`src/backends/apple/README.md`。
</details>

<details>
<summary><strong>Intel CPU / 核显 / Arc</strong> &nbsp;·&nbsp; 测试中 &nbsp;·&nbsp; OpenVINO</summary>

```bash
cmake -S . -B build-intel -DTURBO_BACKENDS="cpu;intel"
cmake --build build-intel -j$(nproc)
./build-intel/turboocr-server --backend intel      # OV_DEVICE=CPU|GPU|NPU
```

在同一块芯片上，原生 OpenVINO 路径快于 ONNX Runtime 路径。详见
`src/backends/intel/README.md`。
</details>

<details>
<summary><strong>AMD GPU（ROCm）</strong> &nbsp;·&nbsp; 尚未在真实硬件上验证</summary>

通过 ROCm 运行：HIP 内核 + MIGraphX 推理引擎，带按架构区分的 `.mxr`
编译缓存，模型编译只需一次。

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTURBO_BACKENDS="cpu;amd" \
      -DCMAKE_HIP_ARCHITECTURES="$(rocminfo | grep -om1 'gfx[0-9a-f]*')" \
      -DCMAKE_PREFIX_PATH=/opt/rocm
cmake --build build -j$(nproc)
./build/turboocr-server --backend amd
```

首次运行会编译 MIGraphX 图并缓存到 `~/.cache/turbo-ocr/mgx_*.mxr`。
首台机器的检查清单见 `src/backends/amd/BRINGUP.md`。
</details>

<details>
<summary><strong>仅 CPU</strong> &nbsp;·&nbsp; 便携后备方案，处处可用</summary>

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DUSE_CPU_ONLY=ON
cmake --build build -j$(nproc)
./build/turboocr-server
```
</details>

首个请求，在任何后端上都相同（本机构建监听 `8080`；上文 Docker 快速开始映射 `8000`）：

```bash
curl -X POST http://localhost:8080/ocr/raw \
  --data-binary @document.png -H "Content-Type: image/png"
```

```json
{"results": [{"text": "Invoice Total", "confidence": 0.97, "bounding_box": [[42,10],[210,10],[210,38],[42,38]]}]}
```

各阶段按请求开启：`?layout=1`、`?tables=1`、`?formulas=1`（表格与公式会自动启用版面）。
PDF 用 `POST /ocr/pdf`，Markdown 导出用 `?markdown=1` 或 `POST /ocr/markdown`，gRPC 在
50051 端口。`GET /capabilities` 报告运行中的服务器加载了哪些能力；请求服务器未启用的
阶段会得到明确的 `400`，绝不会静默返回空结果。

---

## 基准测试

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="tests/benchmark/comparison/images/bench_hero_dark.svg">
    <img src="tests/benchmark/comparison/images/bench_hero_light.svg" alt="单张 RTX 5090 整页 OCR 吞吐：TurboOCR 在 FUNSD 上 678 张/秒，其他引擎 2–6 张/秒" width="100%">
  </picture>
</p>

精度方面：`medium` 档在 FUNSD / CORD 上达到 **91.9% / 93.4%** 词级 F1，为所有被测
引擎中最高；完整结构化解析在 OmniDocBench 上取得 **0.90** Overall，与 PaddleOCR-VL
相差约 5 个百分点，而速度约为其二十倍。所有数字均来自仓库内基准套件，在单张
RTX 5090 上测得：所有引擎使用完全相同的页面，计时窗口 ≥15 秒，双时钟交叉校验。

→ [完整基准与方法论](docs/benchmarks/comparison.md)

---

## 提升精度

默认配置以吞吐为先。三个旋钮用速度换精度：

1. **更大的模型档位** — `-e OCR_MODEL=small`（以约一半速度修复 tiny 的大部分误读）或 `medium`（最准确）。
2. **更高的检测分辨率** — `-e DET_MAX_SIDE_LIMIT=2560`，适合手机截图或密集扫描件上的小字（一次性重建引擎，之后缓存）。
3. **每行方向分类** — `-e CLS_ALL_BOXES=1`，适合混有倒置横排文字的扫描件。

每个旋钮及组合的实测代价：

<p align="center">
  <img src="tests/benchmark/comparison/images/lever_cost.png" alt="各精度旋钮在各模型档位上的实测吞吐代价" width="88%">
</p>

---

## 模型

文本检测 + 识别 + 行方向分类始终运行；其余全部按需加载。

| 阶段 | 模型 / 架构 | 大小 | 启用方式 | 文档 |
|---|---|---:|---|---|
| **文本检测** | PP-OCRv6 det（DB，三档） | 1.7 / 9.4 / 59 MB | `OCR_MODEL` 档位 | [detection](docs/models/detection.md) |
| **文本识别** | PP-OCRv6 rec（拉丁 + 中文 + 日文） | 4.3 / 20 / 73 MB | `OCR_MODEL` 档位 — 默认 `tiny` | [recognition](docs/models/recognition.md) |
| **行方向** | PP-LCNet 每行 0°/180° | ~1 MB | 始终开启（`CLS_ALL_BOXES=1` 检查每一行） | [classification](docs/models/classification.md) |
| **页面方向** | PP-LCNet doc_ori 0/90/180/270 | ~7 MB | `/ocr/pdf?autorotate=1` | [http api](docs/reference/http.md) |
| **版面** | PP-DocLayoutV3（RT-DETR-L，25 类） | ~124 MB | `?layout=1` | [layout](docs/models/layout.md) |
| **表格 → HTML** | SLANet-Plus（TRT 编码器 + C++ GRU 解码器） | ~5 MB | `TABLE_BACKEND=slanext` + `?tables=1` | [table](docs/models/table.md) |
| **公式 → LaTeX** | PP-FormulaNet_plus-S（纯 C++，无 Python） | ~294 MB | `FORMULA_BACKEND=ppformulanet_s` + `?formulas=1` | [formula](docs/models/formula.md) |

档位在速度与精度之间取舍，字符集大体一致，但有一个重要例外：`tiny` 不含日文假名及大部分 CJK 汉字，日文请用 `small` 或 `medium`（见[模型选择](docs/models/selection.md)）。其他文字（`arabic`、`eslav`、`korean`、
`thai`、`greek`）使用保留的 PP-OCRv5 识别器。`tables=1`/`formulas=1` 会自动启用版面；
默认路径不为未使用的阶段付出任何代价。

→ [模型选择指南](docs/models/selection.md)

---

## Python

`python/` 包封装同一条 C++ 流水线（nanobind，推理时释放 GIL），不是 Python 重写。
模型按档位自动下载（`tiny` 约 6 MB），带 SHA256 校验。尚未发布到 PyPI；用
`pip wheel python/` 构建 wheel。

```python
import turboocr

ocr = turboocr.OCR(tier="tiny", replicas=3)   # 内置副本池
page = ocr.read("invoice.png")                # 单图 → PageResult
doc = ocr.read_batch(images)                  # 自动分发到各副本
ocr.read_pdf("report.pdf")                    # PDF → DocumentResult
```

一个 `OCR(replicas=3)` 对象即可达到服务器多副本吞吐（Apple 芯片上实测为其 94%），
无需自行管理线程。`backend=` 可选 `"cuda"`、`"apple"`、`"openvino"`、`"cpu"` 等 —
与服务器共用同一套后端抽象。

→ [python/README.md](python/README.md) · [设计文档](python/DESIGN.md)

---

## API

单个二进制从共享流水线池同时提供 HTTP 与 gRPC。

| 端点 | 用途 |
|---|---|
| `POST /ocr/raw` | OCR 原始图片字节（最快） |
| `POST /ocr` | OCR JSON 中的 base64 图片 |
| `POST /ocr/pixels` | 零解码原始像素缓冲 |
| `POST /ocr/batch` | 批量图片 |
| `POST /ocr/pdf` | PDF → 文本；`?markdown=1` → 整本 PDF 转 Markdown |
| `POST /ocr/markdown` | 单页 → 忠实 Markdown（需要版面） |
| `POST /ocr/stream` | PDF → 逐页 newline-delimited JSON 事件 |
| `POST /infer` | OCR + 版面 / 阅读顺序 / 区块，单次响应 |
| `GET /capabilities` | 运行时能力与路由发现 |
| `GET /metrics` · `/profile` · `/health` | Prometheus · 分阶段耗时 · 探活 |

OCR 与识别类端点在 50051 端口有对应的 gRPC 实现，共用同一套校验核心，两种传输不会漂移（映射关系及少数仅 HTTP 的端点见 [gRPC 参考](docs/reference/grpc.md)）。
所有 OCR 端点均接受 `?layout=1`、`?tables=1`、`?formulas=1`。

→ [HTTP API](docs/reference/http.md) · [gRPC API](docs/reference/grpc.md) · [监控](docs/reference/monitoring.md)

---

## 配置

一切皆环境变量（均有等价 CLI 参数）。常用项：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OCR_MODEL` | `tiny` | `tiny` / `small` / `medium`，或 PP-OCRv5 文字模型 |
| `DISABLE_LAYOUT` | `0` | `1` 跳过版面模型（省 ~300–500 MB 显存） |
| `CLS_ALL_BOXES` | `0` | `1` 对每一行运行 0°/180° 分类器 |
| `DET_MAX_SIDE_LIMIT` | `1280` | 检测分辨率上限（密集扫描件可调高） |
| `REQUEST_TIMEOUT_MS` | `60000` | 排队超时，超过返回 `504` |
| `SHUTDOWN_GRACE_SECONDS` | `30` | SIGTERM 后的真实排空上限 — 超时后排队任务被丢弃，进行中的请求跑完 |
| `PIPELINE_POOL_SIZE` | 自动 | 并发 GPU 流水线数量 |

→ [完整配置参考（35+ 变量）](docs/reference/configuration.md)

---

## 从源码构建

```bash
# Docker（推荐）
docker build -f docker/Dockerfile --target nvidia -t turboocr .

# 原生（首次构建自动拉取模型到 ./models/）
cmake -B build -DTENSORRT_DIR=/usr/local/tensorrt
cmake --build build -j$(nproc)
LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/turboocr-server
```

需要 GCC 13.3+/C++20、CUDA + TensorRT 10.2+、OpenCV 4.x、Drogon 1.9+、gRPC。
Wuffs、Clipper、PDFium 已随仓库内置于 `third_party/`。从 v2.x 升级请先阅读
**[升级到 v3 — 破坏性变更](docs/guides/upgrading-v3.md)**；v4 的新增内容见
**[v4 变更说明](docs/guides/upgrading-v4.md)**。

→ [构建指南与 GPU 架构说明](docs/getting-started/native.md)

---

## 致谢

基于以下开源工作构建：

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**（百度）— PP-OCRv6 / PP-OCRv5 检测、识别与分类模型，以及 PP-DocLayoutV3 版面检测。没有他们的研究与预训练权重，本项目无从谈起。
- **[Drogon](https://drogon.org)** — 高性能异步 C++ HTTP 框架。
- **[Wuffs](https://github.com/google/wuffs)** — Google 出品的快速 PNG 解码器（内置）。
- **[PDFium](https://pdfium.googlesource.com/pdfium/)** — PDF 渲染与文本提取（内置）。
- **[Clipper](http://www.angusj.com/delphi/clipper.php)** — 文本检测后处理的多边形裁剪（内置）。

## 许可证

MIT。见 [LICENSE](LICENSE)。

<p align="center">
  <a href="https://github.com/aiptimizer/TurboOCR"><strong>⭐ 在 GitHub 上给 TurboOCR 点个 Star</strong></a><br>
  <sub>由 <a href="https://miruiq.com"><strong>Miruiq</strong></a>（AI 驱动的 PDF 与文档数据提取）与 <a href="https://diaiq.com"><strong>DiaIQ</strong></a> 赞助。</sub>
</p>
