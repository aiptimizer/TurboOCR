/* TurboOCR install selector — PyTorch-style matrix.
 * Two pill rows (hardware, install method) drive one command card.
 * Initialized through Material's document$ observable so it re-binds after
 * navigation.instant page swaps. */

(function () {
  "use strict";

  var CONFIG = {
    nvidia: {
      label: "NVIDIA GPU",
      status: "shipped",
      methods: {
        docker: {
          command:
            "docker build -f docker/Dockerfile --target nvidia -t turboocr:nvidia .\n" +
            "docker run --gpus all -p 8000:8000 -p 50051:50051 \\\n" +
            "  -v trt-cache:/home/ocr/.cache/turbo-ocr \\\n" +
            "  turboocr:nvidia",
          note:
            "First start builds TensorRT engines (~90 s on a 5090; TRT_OPT_LEVEL=3 cuts it 3–5x on older cards) and caches them in the volume. " +
            "Add stages with -e TABLE_BACKEND=slanext, -e FORMULA_BACKEND=ppformulanet_s, -e OCR_MODEL=medium.",
        },
        source: {
          command:
            "cmake -B build -DTENSORRT_DIR=/usr/local/tensorrt\n" +
            "cmake --build build -j$(nproc)\n" +
            "LD_LIBRARY_PATH=/usr/local/tensorrt/lib ./build/turboocr-server",
          note:
            "Needs GCC 13.3+/C++20, CUDA + TensorRT 10.2+, OpenCV 4.x, Drogon 1.9+, gRPC. Models are auto-fetched into ./models/ on first build.",
        },
        python: {
          command:
            "scripts/python/build_backend_wheel.sh cuda12   # or cuda13\n" +
            "pip install build-wheels/cuda12/fixed/*.whl\n" +
            'python -c "import turboocr_engine; print(turboocr_engine.OCR(backend=\'cuda\').read(\'doc.png\'))"',
          note:
            "Builds the turboocr-engine-cuda12 wheel from this checkout; the helper also repairs it, because a bare pip wheel only runs on the machine that built it. NVIDIA ships TWO wheels, one per CUDA major — cuda12 needs driver R525+, cuda13 needs R580+ — and both carry TensorRT 10.15.1.29. backend=\u0027cuda\u0027 is the instant-start CUDA execution provider; backend=\u0027turbo\u0027 is native TensorRT with a one-time cached engine build. The cuda12/cuda13 wheels are awaiting a PyPI file-size approval; once live: pip install --pre \"turboocr[cuda12]\" (or [cuda13]).",
        },
      },
    },
    apple: {
      label: "Apple Silicon",
      status: "testing",
      noDocker: "macOS containers have no GPU passthrough — the Apple backend runs natively.",
      methods: {
        source: {
          command:
            "brew install cmake opencv drogon jsoncpp protobuf grpc c-ares jpeg-turbo\n" +
            'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBO_BACKENDS="cpu;apple"\n' +
            'cmake --build build -j"$(sysctl -n hw.ncpu)"\n' +
            "./build/turboocr-server --backend apple",
          note:
            "One-time prereqs: full Xcode + Metal toolchain (`xcodebuild -downloadComponent MetalToolchain`) and an osx-arm64 ONNX Runtime ≥ 1.27 — " +
            "see Native build → macOS. Detection runs on Metal + MPSGraph; recognition is a GPU + Neural Engine hybrid (narrow crops on the ANE via CoreML, in parallel). " +
            "TURBO_APPLE_ANE_MAXW tunes the split (default 800).",
        },
        python: {
          command:
            'pip install --pre "turboocr[apple]"\n' +
            'python -c "import turboocr; print(turboocr.OCR(backend=\'apple\', replicas=3).read(\'doc.png\'))"',
          note:
            "Live on PyPI, no toolchain needed (--pre required, pre-release). The macOS arm64 wheel runs the full native mode out of the box — Metal GPU + Neural Engine, with the export bundles auto-downloaded and SHA256-verified on first use; detection adapts to any page shape at runtime, and aread/aread_batch/aread_pdf give asyncio concurrency over the replica pool (replicas=3 measured at 94% of the server's multi-replica throughput). " +
            "To build the same wheel from this checkout instead: scripts/python/build_backend_wheel.sh cpu, then pip install build-wheels/cpu/fixed/*.whl (needs the macOS prereqs above — it compiles the same C++ tree).",
        },
      },
    },
    intel: {
      label: "Intel CPU / iGPU / Arc",
      status: "testing",
      methods: {
        docker: {
          command:
            "docker build -f docker/Dockerfile --target intel -t turboocr:intel .\n" +
            "docker run -p 8000:8000 -p 50051:50051 turboocr:intel",
          note:
            "Built from this repo (no published Intel image yet). The image pins --backend intel for you and defaults to OpenVINO's CPU device, so it runs with no device passthrough. For the iGPU/Arc, pass the device through AND select it: docker run --device /dev/dri -e OV_DEVICE=GPU … (--device /dev/dri alone only makes the hardware visible).",
        },
        source: {
          command:
            'cmake -S . -B build -DTURBO_BACKENDS="cpu;intel"\n' +
            "cmake --build build -j$(nproc)\n" +
            "./build/turboocr-server --backend intel",
          note:
            "-DTURBO_BACKENDS compiles the cpu and intel backends into the one server binary; --backend intel picks which one runs, and it is REQUIRED here — started without it, the server auto-selects the plain CPU path even though you just built the Intel backend. OV_DEVICE=CPU|GPU|NPU picks which Intel silicon OpenVINO uses (default GPU = the iGPU/Arc). The OpenVINO runtime must be on CMAKE_PREFIX_PATH.",
        },
        python: {
          command:
            "scripts/python/build_backend_wheel.sh openvino\n" +
            "pip install build-wheels/openvino/fixed/*.whl\n" +
            'python -c "import turboocr_engine; print(turboocr_engine.OCR(backend=\'openvino\').read(\'doc.png\'))"',
          note: "Builds the turboocr-engine-openvino wheel from this checkout (building needs the OpenVINO dev package; at run time the wheel's own openvino pip dependency supplies the runtime automatically). backend=\x27openvino\x27 runs the native OpenVINO engine; OV_DEVICE=CPU|GPU|NPU or device= picks the device. Published on PyPI: pip install --pre \"turboocr[openvino]\" works today (--pre required, pre-release).",
        },
      },
    },
    amd: {
      label: "AMD GPU",
      status: "not yet hardware-tested",
      methods: {
        docker: {
          command:
            "docker build -f docker/Dockerfile --target amd -t turboocr:amd .\n" +
            "docker run --device /dev/kfd --device /dev/dri --group-add video \\\n" +
            "  -v ocr-cache:/home/ocr/.cache/turbo-ocr \\\n" +
            "  -p 8000:8000 -p 50051:50051 turboocr:amd",
          note:
            "Built from this repo (no published AMD image yet). /dev/kfd + /dev/dri expose the GPU to ROCm inside the container. First run compiles ~42 MIGraphX graphs; the named volume persists that cache so only the first start pays.",
        },
        source: {
          command:
            "cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \\\n" +
            '      -DTURBO_BACKENDS="cpu;amd" \\\n' +
            "      -DCMAKE_HIP_ARCHITECTURES=\"$(rocminfo | grep -om1 'gfx[0-9a-f]*')\" \\\n" +
            "      -DCMAKE_PREFIX_PATH=/opt/rocm\n" +
            "cmake --build build -j$(nproc)\n" +
            "./build/turboocr-server --backend amd",
          note:
            "HIP kernels + MIGraphX engine. The first run compiles the graphs and caches them under ~/.cache/turbo-ocr/mgx_*.mxr; steady state starts instantly. First-machine checklist: src/backends/amd/BRINGUP.md.",
        },
        python: {
          command:
            "scripts/python/build_backend_wheel.sh rocm\n" +
            "pip install build-wheels/rocm/fixed/*.whl\n" +
            'python -c "import turboocr_engine; print(turboocr_engine.OCR(backend=\'rocm\').read(\'doc.png\'))"',
          note: "Builds the turboocr-engine-rocm wheel from this checkout (needs ROCm on the host; compiles clean but not yet validated on AMD hardware). Once the engine wheels reach PyPI this becomes: pip install \"turboocr[rocm]\".",
        },
      },
    },
    cpu: {
      label: "CPU only",
      status: "shipped",
      methods: {
        docker: {
          command:
            "docker build -f docker/Dockerfile --target cpu -t turboocr:cpu .\n" +
            "docker run -p 8000:8000 -p 50051:50051 turboocr:cpu",
          note: "Built from this repo. No devices to pass through — runs anywhere Docker runs.",
        },
        source: {
          command:
            'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBO_BACKENDS="cpu"\n' +
            "cmake --build build -j$(nproc)\n" +
            "./build/turboocr-server",
          note: "ONNX Runtime + OpenCV, no GPU required. Runs anywhere.",
        },
        python: {
          command:
            "scripts/python/build_backend_wheel.sh cpu\n" +
            "pip install build-wheels/cpu/fixed/*.whl\n" +
            'python -c "import turboocr_engine; print(turboocr_engine.OCR().read(\'doc.png\'))"',
          note: "The portable default — works on any machine. Published on PyPI: pip install --pre \"turboocr[cpu]\" works today (--pre required, pre-release).",
        },
      },
    },
  };

  var METHOD_LABELS = { docker: "Docker", source: "Build from source", python: "Python library" };

  function init(root) {
    var state = { hw: "nvidia", method: "docker" };
    var hwRow = root.querySelector('[data-row="hw"]');
    var methodRow = root.querySelector('[data-row="method"]');
    var cmdEl = root.querySelector(".phc-sel-cmd code");
    var noteEl = root.querySelector(".phc-sel-note");
    var statusEl = root.querySelector(".phc-sel-status");
    var copyBtn = root.querySelector(".phc-sel-copy");

    // The skeleton in install.md ships EMPTY on purpose: GitHub renders the
    // markdown but never runs this file, and stray "Hardware"/"Run as"/"copy"
    // text nodes floating above an empty box is what a reader saw there before.
    // The labels belong to the working widget, so the working widget writes them.
    var labels = root.querySelectorAll(".phc-sel-label");
    if (labels[0]) labels[0].textContent = "Hardware";
    if (labels[1]) labels[1].textContent = "Run as";
    copyBtn.textContent = "copy";

    function pill(row, key, label) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "phc-pill";
      b.dataset.key = key;
      b.textContent = label;
      row.appendChild(b);
      return b;
    }

    Object.keys(CONFIG).forEach(function (k) {
      pill(hwRow, k, CONFIG[k].label);
    });
    Object.keys(METHOD_LABELS).forEach(function (k) {
      pill(methodRow, k, METHOD_LABELS[k]);
    });

    function render() {
      var hw = CONFIG[state.hw];
      if (!hw.methods[state.method]) state.method = "source";
      var entry = hw.methods[state.method];

      hwRow.querySelectorAll(".phc-pill").forEach(function (b) {
        b.classList.toggle("phc-pill--on", b.dataset.key === state.hw);
      });
      methodRow.querySelectorAll(".phc-pill").forEach(function (b) {
        var enabled = !!hw.methods[b.dataset.key];
        b.classList.toggle("phc-pill--on", b.dataset.key === state.method);
        b.classList.toggle("phc-pill--off", !enabled);
        b.title = enabled ? "" : hw.noDocker || "";
      });

      cmdEl.textContent = entry.command;
      noteEl.textContent = entry.note;
      statusEl.textContent = hw.status;
      statusEl.dataset.status = hw.status === "shipped" ? "good" : "note";
    }

    hwRow.addEventListener("click", function (e) {
      var b = e.target.closest(".phc-pill");
      if (b) { state.hw = b.dataset.key; render(); }
    });
    methodRow.addEventListener("click", function (e) {
      var b = e.target.closest(".phc-pill");
      if (b && !b.classList.contains("phc-pill--off")) { state.method = b.dataset.key; render(); }
    });
    copyBtn.addEventListener("click", function () {
      navigator.clipboard.writeText(cmdEl.textContent).then(function () {
        copyBtn.textContent = "copied";
        setTimeout(function () { copyBtn.textContent = "copy"; }, 1200);
      });
    });

    render();

    // Only now, with a working selector on screen, retire the static list —
    // it carries the same commands and would otherwise be shown twice. It is
    // hidden LAST so that a throw anywhere above leaves the reader with the
    // static commands rather than an empty box.
    document.querySelectorAll(".phc-static").forEach(function (el) {
      el.hidden = true;
    });
  }

  function boot() {
    var root = document.querySelector(".phc-installer");
    if (root && !root.dataset.ready) {
      root.dataset.ready = "1";
      init(root);
    }
  }

  if (window.document$ && window.document$.subscribe) {
    window.document$.subscribe(boot);
  } else {
    document.addEventListener("DOMContentLoaded", boot);
  }
})();
