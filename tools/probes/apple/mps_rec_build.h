#pragma once
// FORWARDING HEADER — the real translator lives at
// src/backends/apple/engine/mps_rec_build.h (it moved there when it stopped
// being a probe-only header and became library code linked into the Apple
// backend).
//
// The standalone tools/probes/apple/mps_*.mm probes are built with the
// documented recipe
//   clang++ -std=c++20 -ObjC++ -fobjc-arc -O2 -Iinclude -Itools/probes/apple ...
// which has no -Isrc/backends, and they all `#include "mps_rec_build.h"`.
// This one-liner keeps that recipe working without touching twelve probes:
// the path below is relative to THIS file (tools/probes/apple/), so it resolves
// no matter what the -I list or the working directory is.
//
// The Apple backend itself does NOT go through here — src/backends/apple/
// engine/mps_engine.mm includes "apple/engine/mps_rec_build.h" directly off
// -Isrc/backends.
#include "../../../src/backends/apple/engine/mps_rec_build.h"
