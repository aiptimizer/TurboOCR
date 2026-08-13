// Python bindings for the TurboOCR C++ engine (nanobind).
//
// This is a THIN marshaling layer — no pipeline logic lives here. A BGR uint8
// image goes in as a zero-copy ndarray; the real work (detection, recognition,
// CTC, warp, layout/table/formula) runs in the ONE UnifiedOcrPipeline over a
// backend::Backend selected at init ("cpu" by default — the portable ORT
// backend whose execution provider the Python layer selects via env before
// construction: ORT_EP, DET_*, REC_BATCH_N, ...).
//
// The old binding wrapped CpuOcrPipeline directly; that class is gone —
// UnifiedOcrPipeline + the backend seam replaced it. Table/formula stages stay
// env/routing-config driven (TABLE_SLANEXT_ENCODER_ONNX, FORMULA_ONNX +
// FORMULA_TOKENIZER), exactly like the server: the Python layer sets those and
// calls load_structure().

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "turbo_ocr/base/log/stage_profiler.h"

#include <iostream>
#include <map>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include "turbo_ocr/analysis/detection/det_config.h" // set_det_config_base
#include "turbo_ocr/backend/backend.h"
#include "turbo_ocr/serialization/serialization.h"  // assign_layout_ids
#include "turbo_ocr/core/types.h"
#include "turbo_ocr/core/layout_types.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/pipeline/unified/unified_ocr_pipeline.h"
#include "turbo_ocr/core/router_types.h"
// The SAME request-option gate HTTP and gRPC run, adapted to the binding's
// keyword arguments. Python is a transport client like any other: without this
// it reached the pipeline with no availability gate, no reading_order/as_blocks
// implications and no text=0 combination rules.
#include "turbo_ocr/service/validation/python_options.h"

namespace nb = nanobind;
using namespace nb::literals;

using turbo_ocr::OCRResultItem;
using turbo_ocr::pipeline::OcrPipelineResult;
using turbo_ocr::pipeline::RunFlags;
using turbo_ocr::pipeline::UnifiedOcrPipeline;

namespace {

// Wrap a [H,W,3] uint8 C-contiguous ndarray as a BGR cv::Mat WITHOUT copying —
// the Mat borrows the caller's buffer, valid for the duration of the call.
cv::Mat as_bgr(const nb::ndarray<const uint8_t, nb::ndim<3>, nb::c_contig> &a) {
  if (a.shape(2) != 3)
    throw std::invalid_argument("image must be HxWx3 BGR uint8");
  return cv::Mat(static_cast<int>(a.shape(0)), static_cast<int>(a.shape(1)),
                 CV_8UC3, const_cast<uint8_t *>(a.data()));
}

// One recognized line, exposed to Python. `box` is [tl,tr,br,bl] as 4 (x,y) pairs.
struct PyItem {
  std::string text;
  float confidence;
  std::vector<std::array<int, 2>> box;
  std::string source;
  int id;
  int layout_id;
};

PyItem to_py(const OCRResultItem &r) {
  PyItem it;
  it.text = r.text;
  it.confidence = r.confidence;
  it.box = {{{r.box[0][0], r.box[0][1]}},
            {{r.box[1][0], r.box[1][1]}},
            {{r.box[2][0], r.box[2][1]}},
            {{r.box[3][0], r.box[3][1]}}};
  it.source = r.source;
  it.id = r.id;
  it.layout_id = r.layout_id;
  return it;
}

std::vector<PyItem> to_py_list(const std::vector<OCRResultItem> &items) {
  std::vector<PyItem> out;
  out.reserve(items.size());
  for (const auto &r : items)
    out.push_back(to_py(r));
  return out;
}

// One layout region (PP-DocLayoutV3): human label + score + quad + id.
struct PyLayout {
  std::string label;
  float score;
  std::vector<std::array<int, 2>> box;
  int id;
  int parent_id;  // containing region's id, -1 for a top-level region
};

PyLayout to_py_layout(const turbo_ocr::layout::LayoutBox &lb) {
  PyLayout p;
  p.label = std::string(turbo_ocr::layout::label_name(lb.class_id));
  p.score = lb.score;
  p.box = {{{lb.box[0][0], lb.box[0][1]}},
           {{lb.box[1][0], lb.box[1][1]}},
           {{lb.box[2][0], lb.box[2][1]}},
           {{lb.box[3][0], lb.box[3][1]}}};
  p.id = lb.id;
  p.parent_id = lb.parent_id;
  return p;
}

// A recognized table region (SLANeXt → HTML) or formula region (→ LaTeX).
struct PyStructured {
  int layout_id;
  std::string content;  // HTML for tables, LaTeX for formulas
  float score;
  std::vector<std::array<int, 2>> box;
};

template <typename T>
PyStructured to_py_structured(const T &r, const std::string &content) {
  PyStructured p;
  p.layout_id = r.layout_id;
  p.content = content;
  p.score = r.score;
  p.box = {{{r.box[0][0], r.box[0][1]}},
           {{r.box[1][0], r.box[1][1]}},
           {{r.box[2][0], r.box[2][1]}},
           {{r.box[3][0], r.box[3][1]}}};
  return p;
}

// run_with_layout result: text + layout regions + reading order + tables +
// formulas + degradation.
struct PyResult {
  std::vector<PyItem> items;
  std::vector<PyLayout> layout;
  std::vector<int> reading_order;
  std::vector<PyStructured> tables;
  std::vector<PyStructured> formulas;
  bool text_degraded = false;
  std::string text_warning;
  bool table_degraded = false;
  std::string table_warning;
  bool formula_degraded = false;
  std::string formula_warning;
};

// The Python-facing pipeline: one Backend + one UnifiedOcrPipeline replica.
// Single-flight per instance (the Python layer holds a lock around run()),
// matching UnifiedOcrPipeline's one-thread-at-a-time contract.
class PyPipeline {
public:
  // Load detector + recognizer (+ optional angle cls / layout / doc-orientation)
  // through the named backend ("cpu" = the portable ORT backend). Returns false
  // when the backend is not compiled in or a REQUIRED stage fails to load.
  bool init(const std::string &det, const std::string &rec,
            const std::string &dict, const std::string &cls,
            const std::string &layout, const std::string &doc_ori,
            const std::string &backend_name, const std::string &mode,
            bool fp16, const std::string &device) {
    backend_ = turbo_ocr::backend::make_backend(backend_name);
    if (!backend_) return false;
    turbo_ocr::backend::BackendConfig cfg;
    // WHICH PATH to the silicon: "native"/"ultra" (vendor graph engine),
    // "onnx"/"fast" (the .onnx on that vendor's ORT provider, no graph build),
    // "" / "auto" = native when its artefact exists, else onnx.
    cfg.mode = turbo_ocr::backend::parse_engine_mode(mode);
    cfg.ep.fp16 = fp16;
    cfg.ep.device = device;
    cfg.det_model = det;
    cfg.rec_model = rec;
    cfg.rec_dict = dict;
    cfg.cls_model = cls;
    cfg.layout_model = layout;
    cfg.doc_orient_model = doc_ori;
    cfg.want_layout = !layout.empty();
    auto stages = backend_->load_stages(cfg);
    if (!stages.available.detector || !stages.available.recognizer)
      return false;
    loaded_ = stages.available.optional;
    has_layout_ = loaded_.get(turbo_ocr::capability::CapabilityId::Layout);
    pipe_ = std::make_unique<UnifiedOcrPipeline>(*backend_, std::move(stages),
                                                 backend_->make_queue());
    return true;
  }

  // Table/formula bootstrap from the routing config / env (the Python layer
  // sets TABLE_SLANEXT_ENCODER_ONNX / FORMULA_ONNX + FORMULA_TOKENIZER first).
  // Returns false only when an explicitly configured LOCAL backend failed to
  // load — same fail-loud contract as the server.
  bool load_structure() {
    ensure();
    return pipe_->load_router_models() && pipe_->load_table_backend();
  }

  bool has_layout() const { return has_layout_; }

  // LOADED (capability/capability.h) as a {name: bool} dict, keyed by the SAME
  // wire names the HTTP query params and /capabilities use. Built by iterating
  // the capability table, so a new capability appears in Python without
  // touching this binding — the per-capability has_*() accessors below cannot
  // make that claim, which is why they are kept only for compatibility.
  std::map<std::string, bool> capabilities() const {
    const auto live = live_capabilities();
    std::map<std::string, bool> out;
    for (const auto &cap : turbo_ocr::capability::kCapabilities)
      out.emplace(std::string(cap.name), live.get(cap.id));
    return out;
  }
  bool has_table_backend() const {
    return pipe_ && pipe_->has_default_table_backend();
  }
  bool has_formula_backend() const {
    return pipe_ && pipe_->has_default_formula_backend();
  }
  bool has_doc_ori() const { return pipe_ && pipe_->has_doc_ori(); }

  // The mode the backend ACTUALLY came up on — an "auto" request that fell
  // back from native to onnx must report onnx, or the Python `info()` lies.
  std::string mode() const {
    if (!backend_) return {};
    return std::string(turbo_ocr::backend::engine_mode_name(backend_->caps().mode));
  }

  void warmup() {
    ensure();
    pipe_->warmup();
  }

  int detect_orientation(const cv::Mat &bgr) {
    ensure();
    return pipe_->detect_orientation(bgr);
  }

  std::vector<OCRResultItem> run(const cv::Mat &bgr) {
    ensure();
    return pipe_->run(bgr);
  }

  // Whole-batch entry point. Without this the Python layer could only ever
  // reach `run()`, so `OCR.read_batch()` was a serial loop submitting one
  // image at a time — the exact "no caller can reach run_batch" problem
  // stage_batcher.h describes, but on the library side. Going through
  // UnifiedOcrPipeline::run_batch lets detection see the whole group.
  std::vector<std::vector<OCRResultItem>>
  run_batch(const std::vector<cv::Mat> &bgrs) {
    ensure();
    return pipe_->run_batch(bgrs);
  }

  // Takes the pipeline's own flag POD rather than four bools of its own: the
  // Python-visible signature (img, layout, reading_order, tables, formulas) is
  // unchanged, but the nanobind lambda now names each flag when it builds this
  // argument, so the binding cannot transpose them on the way down.
  OcrPipelineResult run_with_layout(const cv::Mat &bgr, const RunFlags &flags) {
    ensure();
    return pipe_->run_with_layout(bgr, flags, /*routing=*/{},
                                  /*defer_external=*/false);
  }

  // Validate a request's flags through the SHARED gate every transport runs
  // (validation/options_core.h, reached here via the python adapter) and return
  // the pipeline flags it projects. Throws with the core's own message, so an
  // invalid combination reads identically from Python, HTTP and gRPC.
  //
  // `acts_on` keeps its default (everything but DocOrientation): autorotate is
  // not a run_with_layout flag on this binding either — Python calls
  // detect_orientation() and rotates before submitting, exactly as the HTTP
  // image routes do.
  RunFlags parse_flags(const std::map<std::string, bool> &flags) const {
    turbo_ocr::server::InferOptions opts;
    const auto r = turbo_ocr::server::parse_python_options(
        flags, live_capabilities(), &opts);
    if (!r.error.empty()) throw std::invalid_argument(r.error);
    return RunFlags{.layout = opts.want_layout,
                    .reading_order = opts.want_reading_order,
                    .tables = opts.want_tables,
                    .formulas = opts.want_formulas,
                    .text = opts.want_text};
  }

private:
  void ensure() const {
    if (!pipe_)
      throw std::runtime_error("pipeline not initialized — call init() first");
  }

  // What this pipeline can ACTUALLY run right now. Table/formula load AFTER
  // init() (load_structure()), so the init-time snapshot alone reported them
  // false forever, even after load_structure() succeeded — and the availability
  // gate reads this, so a stale answer would reject a request the pipeline can
  // serve.
  turbo_ocr::capability::CapabilityMask live_capabilities() const {
    auto live = loaded_;
    if (pipe_) {
      live.set(turbo_ocr::capability::CapabilityId::Table,
               pipe_->has_default_table_backend());
      live.set(turbo_ocr::capability::CapabilityId::Formula,
               pipe_->has_default_formula_backend());
    }
    return live;
  }

  // DECLARATION ORDER IS LOAD-BEARING: pipe_ holds a reference to *backend_,
  // so it must be destroyed first (members destroy in reverse order).
  std::unique_ptr<turbo_ocr::backend::Backend> backend_;
  std::unique_ptr<UnifiedOcrPipeline> pipe_;
  turbo_ocr::capability::CapabilityMask loaded_;
  bool has_layout_ = false;
};

}  // namespace

NB_MODULE(_turboocr, m) {
  m.doc() = "Native TurboOCR pipeline (nanobind bindings over the C++ engine)";
  m.attr("__version__") = "0.2.0";

  // The engine prints a few load banners via std::cout. Make cout unbuffered so
  // those writes hit fd 1 immediately — that lets the Python layer's fd-level
  // stdout redirect (quiet_stdout) actually capture them instead of them
  // flushing after the redirect window closes.
  std::cout << std::unitbuf;

  // Real capabilities of THIS build: the ONNX Runtime version + execution
  // providers compiled into the linked ORT, plus the device backends compiled
  // into this module (backend seam), so `doctor` reflects the native path.
  // Install the per-model detection base (det_config.h set_det_config_base).
  // The Python catalog carries each tier's official det config (tiny's
  // box_thresh 0.40) exactly like the C++ registry; without this call the
  // native stages loaded with the 0.45 default and the default OCR() run
  // mis-thresholded the detector relative to the server. pipeline.py calls it
  // with the resolved catalog row BEFORE constructing a Pipeline. Env DET_*
  // overrides still win (read_* layers them on top).
  m.def(
      "set_det_config",
      [](const std::string &limit_type, int limit_side_len, int max_side_limit,
         float thresh, float box_thresh, float unclip_ratio) {
        static std::string stored_limit_type; // DetResizeParams keeps a char*
        stored_limit_type = limit_type;
        turbo_ocr::detection::set_det_config_base(
            {stored_limit_type.c_str(), limit_side_len, max_side_limit},
            {thresh, box_thresh, unclip_ratio});
      },
      nb::arg("limit_type"), nb::arg("limit_side_len"),
      nb::arg("max_side_limit"), nb::arg("thresh"), nb::arg("box_thresh"),
      nb::arg("unclip_ratio"));

  // PROFILE_STAGES=1 accumulates per-stage wall time in the native layer;
  // this returns the JSON snapshot and resets the counters — the Python twin
  // of the server's /profile route, for diagnosing where a read() goes.
  m.def("profile_dump", []() { return turbo_ocr::prof::dump_json_and_reset(); });

  m.def("build_info", []() {
    nb::dict d;
    d["ort_version"] = Ort::GetVersionString();
    std::vector<std::string> providers = Ort::GetAvailableProviders();
    d["providers"] = providers;
    std::vector<std::string> backends;
    for (auto b : turbo_ocr::backend::available_backends())
      backends.emplace_back(b);
    d["backends"] = backends;
    return d;
  });

  nb::class_<PyItem>(m, "Item")
      .def_ro("text", &PyItem::text)
      .def_ro("confidence", &PyItem::confidence)
      .def_ro("box", &PyItem::box)
      .def_ro("source", &PyItem::source)
      .def_ro("id", &PyItem::id)
      .def_ro("layout_id", &PyItem::layout_id)
      .def("__repr__", [](const PyItem &i) {
        return "<Item conf=" + std::to_string(i.confidence) + " '" + i.text + "'>";
      });

  nb::class_<PyLayout>(m, "LayoutRegion")
      .def_ro("label", &PyLayout::label)
      .def_ro("score", &PyLayout::score)
      .def_ro("box", &PyLayout::box)
      .def_ro("id", &PyLayout::id)
      .def_ro("parent_id", &PyLayout::parent_id)
      .def("__repr__", [](const PyLayout &l) {
        return "<LayoutRegion " + l.label + " " + std::to_string(l.score) + ">";
      });

  nb::class_<PyStructured>(m, "StructuredRegion")
      .def_ro("layout_id", &PyStructured::layout_id)
      .def_ro("content", &PyStructured::content)
      .def_ro("score", &PyStructured::score)
      .def_ro("box", &PyStructured::box);

  nb::class_<PyResult>(m, "LayoutResult")
      .def_ro("items", &PyResult::items)
      .def_ro("layout", &PyResult::layout)
      .def_ro("reading_order", &PyResult::reading_order)
      .def_ro("tables", &PyResult::tables)
      .def_ro("formulas", &PyResult::formulas)
      .def_ro("text_degraded", &PyResult::text_degraded)
      .def_ro("text_warning", &PyResult::text_warning)
      .def_ro("table_degraded", &PyResult::table_degraded)
      .def_ro("table_warning", &PyResult::table_warning)
      .def_ro("formula_degraded", &PyResult::formula_degraded)
      .def_ro("formula_warning", &PyResult::formula_warning);

  nb::class_<PyPipeline>(m, "Pipeline")
      .def(nb::init<>())
      .def(
          "init",
          [](PyPipeline &p, const std::string &det, const std::string &rec,
             const std::string &dict, const std::string &cls,
             const std::string &layout, const std::string &doc_ori,
             const std::string &backend, const std::string &mode, bool fp16,
             const std::string &device) {
            nb::gil_scoped_release nogil;
            return p.init(det, rec, dict, cls, layout, doc_ori, backend, mode,
                          fp16, device);
          },
          "det"_a, "rec"_a, "dict"_a, "cls"_a = std::string(),
          "layout"_a = std::string(), "doc_ori"_a = std::string(),
          "backend"_a = std::string("cpu"), "mode"_a = std::string("auto"),
          "fp16"_a = true, "device"_a = std::string(),
          "Load detector + recognizer (+ optional angle classifier, layout "
          "model, doc-orientation model) on the named backend, via engine mode "
          "'native'/'ultra', 'onnx'/'fast', or 'auto'. Returns True on success.")
      .def(
          "load_structure",
          [](PyPipeline &p) {
            nb::gil_scoped_release nogil;
            return p.load_structure();
          },
          "Bootstrap table/formula recognizers from the routing config / env "
          "(TABLE_SLANEXT_ENCODER_ONNX, FORMULA_ONNX + FORMULA_TOKENIZER). "
          "Returns False when an explicitly configured local backend failed.")
      .def("capabilities", &PyPipeline::capabilities,
           "What this pipeline loaded, as {capability_name: bool} — the same "
           "names the HTTP API uses (layout/tables/formulas/autorotate).")
      .def("has_layout", &PyPipeline::has_layout)
      .def("has_table_backend", &PyPipeline::has_table_backend)
      .def("has_formula_backend", &PyPipeline::has_formula_backend)
      .def("has_doc_ori", &PyPipeline::has_doc_ori)
      .def("mode", &PyPipeline::mode,
           "The engine mode actually in use: 'native' or 'onnx'.")
      .def(
          "detect_orientation",
          [](PyPipeline &p,
             nb::ndarray<const uint8_t, nb::ndim<3>, nb::c_contig> img) {
            cv::Mat bgr = as_bgr(img);
            nb::gil_scoped_release nogil;
            return p.detect_orientation(bgr);
          },
          "img"_a)
      .def("warmup",
           [](PyPipeline &p) {
             nb::gil_scoped_release nogil;
             p.warmup();
           })
      .def(
          "run",
          [](PyPipeline &p,
             nb::ndarray<const uint8_t, nb::ndim<3>, nb::c_contig> img) {
            cv::Mat bgr = as_bgr(img);
            std::vector<OCRResultItem> items;
            {
              nb::gil_scoped_release nogil;  // let Python threads run during C++
              items = p.run(bgr);
            }
            return to_py_list(items);
          },
          "img"_a, "Run OCR on a BGR uint8 HxWx3 image; returns list[Item].")
      .def(
          "run_batch",
          [](PyPipeline &p,
             const std::vector<
                 nb::ndarray<const uint8_t, nb::ndim<3>, nb::c_contig>> &imgs) {
            // as_bgr returns a VIEW over each array's buffer, so the ndarray
            // handles (owned by `imgs`) must outlive the call — they do: `imgs`
            // is alive for the whole lambda, GIL-released section included.
            std::vector<cv::Mat> bgrs;
            bgrs.reserve(imgs.size());
            for (const auto &img : imgs)
              bgrs.push_back(as_bgr(img));
            std::vector<std::vector<OCRResultItem>> batch;
            {
              nb::gil_scoped_release nogil;
              batch = p.run_batch(bgrs);
            }
            std::vector<std::vector<PyItem>> out;
            out.reserve(batch.size());
            for (const auto &items : batch)
              out.push_back(to_py_list(items));
            return out;
          },
          "imgs"_a,
          "Run a whole batch of HxWx3 BGR uint8 images through one pipeline "
          "submission. Returns one result list per image, in input order.")
      .def(
          "run_with_layout",
          [](PyPipeline &p,
             nb::ndarray<const uint8_t, nb::ndim<3>, nb::c_contig> img,
             bool want_layout, bool want_reading_order, bool want_tables,
             bool want_formulas, bool want_text) {
            cv::Mat bgr = as_bgr(img);
            // Gate FIRST, holding the GIL, so a rejection is a Python exception
            // raised before any work starts. The flags are keyed by their
            // capability-registry names — the same strings the HTTP query
            // params and the proto fields use — which is what makes all three
            // transports share one policy instead of three copies of it.
            // `as_blocks` is deliberately absent: this binding does not
            // aggregate blocks, and offering a flag it would silently ignore is
            // the failure the shared gate exists to prevent.
            const RunFlags flags = p.parse_flags({
                {"layout", want_layout},
                {"reading_order", want_reading_order},
                {"tables", want_tables},
                {"formulas", want_formulas},
                {"text", want_text},
            });
            OcrPipelineResult r;
            {
              nb::gil_scoped_release nogil;
              r = p.run_with_layout(bgr, flags);
              // The local SLANeXt/PP-FormulaNet backends run synchronously, so
              // r.tables/r.formulas are already filled and r.pending is empty
              // (finalize_deferred is only for deferred remote-VLM backends,
              // which this offline binding stubs out).
            }
            // Number the regions and cross-reference each text item to the
            // region containing it, exactly as the HTTP emitters do. Without
            // this, want_reading_order=False handed Python back layout
            // regions with id == -1 (and items with layout_id == -1), which
            // also makes the parent_id links unresolvable. Idempotent: a no-op
            // when the reading-order path already ran it.
            turbo_ocr::assign_layout_ids(r.results, r.layout);
            PyResult out;
            out.items = to_py_list(r.results);
            out.layout.reserve(r.layout.size());
            for (const auto &lb : r.layout)
              out.layout.push_back(to_py_layout(lb));
            out.reading_order = r.reading_order;
            for (const auto &t : r.tables)
              out.tables.push_back(to_py_structured(t, t.html));
            for (const auto &f : r.formulas)
              out.formulas.push_back(to_py_structured(f, f.latex));
            out.text_degraded = r.text_degraded;
            out.text_warning = r.text_warning;
            out.table_degraded = r.table_degraded;
            out.table_warning = r.table_warning;
            out.formula_degraded = r.formula_degraded;
            out.formula_warning = r.formula_warning;
            return out;
          },
          "img"_a, "layout"_a = true, "reading_order"_a = false,
          "tables"_a = false, "formulas"_a = false, "text"_a = true,
          "Run OCR + layout (+ optional tables→HTML / formulas→LaTeX); returns a "
          "LayoutResult. Flags run through the shared request-option gate "
          "(validation/options_core.h): a capability this pipeline did not load, "
          "or an invalid combination, raises ValueError with the same message "
          "the HTTP and gRPC surfaces return.");
}
