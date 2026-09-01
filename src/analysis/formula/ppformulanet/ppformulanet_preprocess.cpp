#include "turbo_ocr/analysis/formula/ppformulanet/ppformulanet_preprocess.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

namespace turbo_ocr::formula {

namespace {
constexpr int S = kFormulaInputSize;
constexpr float MEAN = kFormulaMean, STD = kFormulaStd, PAD_VAL = kFormulaPadVal;

// PIL.Image.resize(BICUBIC) — a SEPARABLE scaled-bicubic resample with antialiasing
// on downscale (filter support grows with the downsample factor). cv2.INTER_CUBIC
// does NOT antialias, so for the wide->short formula crops it diverges from the
// reference by ~1.6/255, wrecking the encoder input. This is a BIT-EXACT port of
// PIL's algorithm (Resample.c): double coeffs (a=-0.5, support 2.0, filterscale=
// max(1,in/out)) quantized to 22-bit fixed point, integer convolution, two uint8
// passes. Float weights drifted ~0.02/px and the greedy decode amplified it.
constexpr int PREC = 22;                        // PIL PRECISION_BITS = 32 - 8 - 2
constexpr int64_t HALF = (int64_t)1 << (PREC - 1);

inline double pil_cubic(double x) {
  const double a = -0.5; x = std::fabs(x);
  if (x < 1.0) return ((a + 2) * x - (a + 3)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

// Precomputed separable filter for one axis (`in_len`->`out_len`), mirroring PIL's
// precompute_coeffs + normalize_coeffs_8bpc. For each output index we store the first
// source tap `lo`, the tap count `n`, and `n` fixed-point weights at row `o*ksize`.
// Precomputing once (vs. per-pixel std::vector churn) lets the convolution be reordered
// for cache locality; the integer weights are byte-for-byte what the old path produced.
struct AxisFilter {
  int ksize = 0;                                // coeff row stride (>= max taps)
  std::vector<int> lo, n;                       // per-output-index tap origin / count
  std::vector<int64_t> coeff;                   // [out_len * ksize] fixed-point weights
  std::vector<double> scratch;                  // reused normalization workspace

  const int64_t *row(int o) const { return &coeff[(size_t)o * ksize]; }
};

void pil_axis(int out_len, int in_len, AxisFilter &f) {
  const double scale = (double)in_len / out_len;
  const double fscale = std::max(1.0, scale), support = 2.0 * fscale;
  f.ksize = (int)std::ceil(support) * 2 + 1;
  f.lo.resize(out_len);
  f.n.resize(out_len);
  f.coeff.assign((size_t)out_len * f.ksize, 0);
  f.scratch.resize(f.ksize);
  for (int xx = 0; xx < out_len; ++xx) {
    const double center = (xx + 0.5) * scale;
    int lo = (int)(center - support + 0.5); if (lo < 0) lo = 0;
    int hi = (int)(center + support + 0.5); if (hi > in_len) hi = in_len;
    const int n = hi - lo;
    double sum = 0.0;
    for (int k = 0; k < n; ++k) { double w = pil_cubic((lo + k - center + 0.5) / fscale); f.scratch[k] = w; sum += w; }
    int64_t *w = &f.coeff[(size_t)xx * f.ksize];
    // PIL divides only when the weight sum is nonzero (cannot happen for the
    // cubic kernel at these supports, but guard the literal port anyway —
    // an unguarded 0/0 would quantize NaN into the coeff table).
    if (sum == 0.0) sum = 1.0;
    for (int k = 0; k < n; ++k) {                          // normalize then quantize
      double v = f.scratch[k] / sum;
      w[k] = (int64_t)(v >= 0 ? v * (1 << PREC) + 0.5 : v * (1 << PREC) - 0.5);
    }
    f.lo[xx] = lo;
    f.n[xx] = n;
  }
}

inline uint8_t clip8(int64_t ss) { ss >>= PREC; return (uint8_t)(ss < 0 ? 0 : ss > 255 ? 255 : ss); }

// Per-thread scratch: formula_preprocess_one runs under `omp parallel for` in both
// recognizer paths, so reuse must be thread_local to stay race-free. Reused across
// calls to drop the per-crop heap traffic (coeff tables, resize planes).
struct ResizeScratch {
  AxisFilter hx, vy;
  cv::Mat tmp, out;
  std::vector<const uint8_t *> rows;            // hoisted source-row pointers (vertical pass)
};

// PIL-bicubic resize of a BGR8 crop -> CV_8UC3, two separable uint8 passes with PIL's
// exact integer convolution (ss = 2^21 + Σ pixel*coeff; out = clip8(ss>>22)).
// WITHIN each pass, integer add/mul are exact and associative, so tap/loop order is
// free (the row-hot horizontal loop below is a pure perf choice). The PASS ORDER is
// NOT free: the intermediate is quantized to uint8 between passes, so horizontal-
// then-vertical (PIL's order) is load-bearing — swapping passes changes the output.
const cv::Mat &pil_resize(const cv::Mat &src, int nw, int nh, ResizeScratch &s) {
  const int sh = src.rows, sw = src.cols;
  pil_axis(nw, sw, s.hx);                        // horizontal: sw -> nw
  s.tmp.create(sh, nw, CV_8UC3);
  for (int y = 0; y < sh; ++y) {                 // y-outer keeps each source row hot
    const uint8_t *srow = src.ptr<uint8_t>(y);
    uint8_t *orow = s.tmp.ptr<uint8_t>(y);
    for (int xx = 0; xx < nw; ++xx) {
      const int n = s.hx.n[xx];
      const int64_t *w = s.hx.row(xx);
      const uint8_t *p = srow + (size_t)s.hx.lo[xx] * 3;
      int64_t b = HALF, g = HALF, r = HALF;
      for (int k = 0; k < n; ++k, p += 3) { b += w[k] * p[0]; g += w[k] * p[1]; r += w[k] * p[2]; }
      uint8_t *o = orow + xx * 3; o[0] = clip8(b); o[1] = clip8(g); o[2] = clip8(r);
    }
  }
  pil_axis(nh, sh, s.vy);                        // vertical: sh -> nh
  s.out.create(nh, nw, CV_8UC3);
  for (int yy = 0; yy < nh; ++yy) {
    const int lo = s.vy.lo[yy], n = s.vy.n[yy];
    const int64_t *w = s.vy.row(yy);
    s.rows.resize(n);
    for (int k = 0; k < n; ++k) s.rows[k] = s.tmp.ptr<uint8_t>(lo + k);
    uint8_t *orow = s.out.ptr<uint8_t>(yy);
    for (int x = 0; x < nw; ++x) {
      const int off = x * 3;
      int64_t b = HALF, g = HALF, r = HALF;
      for (int k = 0; k < n; ++k) { const uint8_t *p = s.rows[k] + off; b += w[k] * p[0]; g += w[k] * p[1]; r += w[k] * p[2]; }
      uint8_t *o = orow + off; o[0] = clip8(b); o[1] = clip8(g); o[2] = clip8(r);
    }
  }
  return s.out;
}
}  // namespace

// Mirror the Python sidecar preprocess: margin-crop -> AR-preserving PIL-bicubic
// resize (longer side 384) -> /255, (x-mean)/std on BGR floats -> BGR2GRAY ->
// center-pad to 384 with the normalized-zero pad value.
void formula_preprocess_one(const uint8_t *bgr, int w, int h, float *out) {
  if (w <= 0 || h <= 0) { std::fill(out, out + S * S, PAD_VAL); return; }
  thread_local ResizeScratch scratch;
  thread_local cv::Mat gray, dataf, mask, bgrf, grayf, padded;

  cv::Mat src(h, w, CV_8UC3, const_cast<uint8_t *>(bgr));
  cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  double mn, mx; cv::minMaxLoc(gray, &mn, &mx);
  cv::Mat cropped = src;
  if (mx > mn) {
    gray.convertTo(dataf, CV_32F);
    dataf = (dataf - mn) / std::max(1.0, mx - mn) * 255.0;
    mask = dataf < 200.0f;  // CV_8U 255/0
    std::vector<cv::Point> nz; cv::findNonZero(mask, nz);
    if (!nz.empty()) {
      cv::Rect r = cv::boundingRect(nz);
      if (r.width > 0 && r.height > 0) cropped = src(r);
    }
  }
  // Python round() is banker's rounding (ties-to-even); std::lrint in the default
  // FE_TONEAREST mode matches it (std::lround would tie-away and shift a side by 1px).
  int W = cropped.cols, Hh = cropped.rows, nw, nh;
  if (W <= Hh) { nw = S; nh = std::max(1, (int)std::lrint((double)S * Hh / W)); }
  else         { nh = S; nw = std::max(1, (int)std::lrint((double)S * W / Hh)); }
  if (nh > S) { nw = std::max(1, (int)std::lrint((double)nw * S / nh)); nh = S; }
  if (nw > S) { nh = std::max(1, (int)std::lrint((double)nh * S / nw)); nw = S; }
  const cv::Mat &resized = pil_resize(cropped, nw, nh, scratch);  // CV_8UC3, PIL-bicubic
  // One convertTo, not a MatExpr: `bgrf = (bgrf - MEAN) / STD` made OpenCV
  // warn "processing of multi-channel arrays might be changed in the future"
  // (matrix_expressions.cpp) on stderr once per process — unsilenceable noise
  // from a library. MEAN/STD are uniform scalars, so the whole chain is one
  // affine map: raw/255 then (x-MEAN)/STD == x*(1/STD) + (-MEAN/STD).
  // Verified bit-identical on the formula corpus, and it drops a full
  // temporary pass over the image.
  resized.convertTo(bgrf, CV_32F, 1.0 / 255.0);
  bgrf.convertTo(bgrf, CV_32F, 1.0 / STD, -MEAN / STD);
  cv::cvtColor(bgrf, grayf, cv::COLOR_BGR2GRAY);
  int dh = S - nh, dw = S - nw, top = dh / 2, bot = dh - top, left = dw / 2, right = dw - left;
  cv::copyMakeBorder(grayf, padded, top, bot, left, right, cv::BORDER_CONSTANT, cv::Scalar(PAD_VAL));
  if (padded.isContinuous()) std::memcpy(out, padded.ptr<float>(), (size_t)S * S * sizeof(float));
  else for (int r = 0; r < S; ++r) std::memcpy(out + (size_t)r * S, padded.ptr<float>(r), (size_t)S * sizeof(float));
}

// Mirror the sidecar's _is_mode_collapsed: the 3-gram uniqueness test catches a
// repetitive runaway run without dropping a legitimate long matrix.
bool formula_is_mode_collapsed(const std::vector<int64_t> &toks, const std::string &latex) {
  if (toks.size() >= 240) return true;
  if (latex.size() >= 1500) return true;
  if (toks.size() >= 50) {
    std::set<std::array<int64_t, 3>> uniq;
    const int n = static_cast<int>(toks.size()) - 2;
    for (int i = 0; i < n; ++i) uniq.insert({toks[i], toks[i + 1], toks[i + 2]});
    if (n > 0 && static_cast<double>(uniq.size()) / n < 0.25) return true;
  }
  return false;
}

}  // namespace turbo_ocr::formula
