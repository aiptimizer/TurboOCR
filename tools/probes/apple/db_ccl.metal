#include <metal_stdlib>
using namespace metal;

// GPU DB post-processing (connected-components) so detection boxes are extracted
// on-GPU, no host round-trip. Label-equivalence propagation: each foreground
// pixel adopts the min label among its 4-neighbours; iterated to convergence.
// Root label = smallest linear pixel index in the component.

kernel void db_init(device const float* prob   [[buffer(0)]],
                    device uint*        label  [[buffer(1)]],
                    constant uint2&     dims   [[buffer(2)]],   // (W,H)
                    constant float&     thresh [[buffer(3)]],
                    uint2 gid [[thread_position_in_grid]])
{
    uint W=dims.x, H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x;
    label[i] = (prob[i] > thresh) ? (i+1u) : 0u;   // 0 = background, else index+1
}

// One propagation sweep; sets *changed if any label decreased.
kernel void db_propagate(device uint*         label   [[buffer(0)]],
                         constant uint2&      dims    [[buffer(1)]],
                         device atomic_uint*  changed [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]])
{
    uint W=dims.x, H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x; uint l=label[i]; if(l==0u) return;
    uint m=l;
    if(gid.x>0u)     { uint n=label[i-1]; if(n!=0u && n<m) m=n; }
    if(gid.x<W-1u)   { uint n=label[i+1]; if(n!=0u && n<m) m=n; }
    if(gid.y>0u)     { uint n=label[i-W]; if(n!=0u && n<m) m=n; }
    if(gid.y<H-1u)   { uint n=label[i+W]; if(n!=0u && n<m) m=n; }
    if(m<l){ label[i]=m; atomic_store_explicit(changed,1u,memory_order_relaxed); }
}

// Run-based scan: ONE thread sweeps a whole row, propagating the running min
// label across each contiguous foreground run (both directions). A horizontal
// text line thus adopts one label in a SINGLE pass instead of O(width) neighbour
// steps. Alternated with the column scan below, CCL converges in a handful of
// passes (not O(diameter)).
kernel void db_rowscan(device uint*        label   [[buffer(0)]],
                       constant uint2&     dims    [[buffer(1)]],
                       device atomic_uint* changed [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    uint W=dims.x, H=dims.y; if(gid>=H) return; uint base=gid*W;
    uint cur=0u;
    for(uint x=0;x<W;x++){ uint l=label[base+x]; if(l==0u){cur=0u;continue;}
        if(cur==0u||l<cur) cur=l; if(cur<l){ label[base+x]=cur; atomic_store_explicit(changed,1u,memory_order_relaxed);} }
    cur=0u;
    for(int x=int(W)-1;x>=0;x--){ uint l=label[base+uint(x)]; if(l==0u){cur=0u;continue;}
        if(cur==0u||l<cur) cur=l; if(cur<l){ label[base+uint(x)]=cur; atomic_store_explicit(changed,1u,memory_order_relaxed);} }
}
kernel void db_colscan(device uint*        label   [[buffer(0)]],
                       constant uint2&     dims    [[buffer(1)]],
                       device atomic_uint* changed [[buffer(2)]],
                       uint gid [[thread_position_in_grid]])
{
    uint W=dims.x, H=dims.y; if(gid>=W) return;
    uint cur=0u;
    for(uint y=0;y<H;y++){ uint i=y*W+gid; uint l=label[i]; if(l==0u){cur=0u;continue;}
        if(cur==0u||l<cur) cur=l; if(cur<l){ label[i]=cur; atomic_store_explicit(changed,1u,memory_order_relaxed);} }
    cur=0u;
    for(int y=int(H)-1;y>=0;y--){ uint i=uint(y)*W+gid; uint l=label[i]; if(l==0u){cur=0u;continue;}
        if(cur==0u||l<cur) cur=l; if(cur<l){ label[i]=cur; atomic_store_explicit(changed,1u,memory_order_relaxed);} }
}

// Path compression (pointer jumping): each pixel's label points at another
// pixel (union-find parent); follow it toward the root so chains collapse in
// O(log diameter) passes instead of O(diameter). Root r has label[r]==r+1.
kernel void db_compress(device uint*        label   [[buffer(0)]],
                        constant uint2&     dims    [[buffer(1)]],
                        device atomic_uint* changed [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]])
{
    uint W=dims.x, H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x; uint l=label[i]; if(l==0u) return;
    uint p=label[l-1u];                 // parent's label
    if(p!=0u && p!=l){ label[i]=p; atomic_store_explicit(changed,1u,memory_order_relaxed); }
}

// Per-component accumulation keyed by root label (index+1). bbox arrays are sized
// W*H (indexed by root-1); atomically min/max the extent and count the pixels.
// xmin/ymin start at UINT_MAX, xmax/ymax at 0, cnt at 0 (host/GPU pre-clears).
kernel void db_bbox(device const uint*   label [[buffer(0)]],
                    constant uint2&      dims  [[buffer(1)]],
                    device atomic_uint*  xmin  [[buffer(2)]],
                    device atomic_uint*  ymin  [[buffer(3)]],
                    device atomic_uint*  xmax  [[buffer(4)]],
                    device atomic_uint*  ymax  [[buffer(5)]],
                    device atomic_uint*  cnt   [[buffer(6)]],
                    device const float*  prob  [[buffer(7)]],
                    device atomic_uint*  psum  [[buffer(8)]],   // fixed-point prob sum (x1024)
                    uint2 gid [[thread_position_in_grid]])
{
    uint W=dims.x, H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x; uint l=label[i]; if(l==0u) return;
    uint r=l-1u;
    atomic_fetch_min_explicit(&xmin[r], gid.x, memory_order_relaxed);
    atomic_fetch_min_explicit(&ymin[r], gid.y, memory_order_relaxed);
    atomic_fetch_max_explicit(&xmax[r], gid.x, memory_order_relaxed);
    atomic_fetch_max_explicit(&ymax[r], gid.y, memory_order_relaxed);
    atomic_fetch_add_explicit(&cnt[r], 1u, memory_order_relaxed);
    atomic_fetch_add_explicit(&psum[r], uint(prob[i]*1024.0f), memory_order_relaxed);
}

struct Box4 { uint xmin, ymin, xmax, ymax; };

// Compact non-empty components (area >= minArea) into a dense box list with an
// atomic count — replaces the host scan, keeping extraction on-GPU.
kernel void db_compact(device const uint* xmin [[buffer(0)]],
                       device const uint* ymin [[buffer(1)]],
                       device const uint* xmax [[buffer(2)]],
                       device const uint* ymax [[buffer(3)]],
                       device const uint* cnt  [[buffer(4)]],
                       constant uint2&    dims [[buffer(5)]],
                       constant uint2&    lim  [[buffer(6)]],   // (minArea, maxBox)
                       device Box4*        boxes    [[buffer(7)]],
                       device atomic_uint* boxCount [[buffer(8)]],
                       uint gid [[thread_position_in_grid]])
{
    uint N=dims.x*dims.y; if(gid>=N) return;
    if(cnt[gid]==0u) return;
    uint w=xmax[gid]-xmin[gid]+1u, h=ymax[gid]-ymin[gid]+1u;
    if(w*h < lim.x) return;
    uint idx=atomic_fetch_add_explicit(boxCount,1u,memory_order_relaxed);
    if(idx>=lim.y) return;
    boxes[idx]=(Box4){xmin[gid],ymin[gid],xmax[gid],ymax[gid]};
}

// Per-box inverse homography for the warp: axis-aligned box (det space) ->
// original-image crop, expanded by `margin` (approximates DB unclip), aspect-
// preserving content width. Threads past the live box count emit width 0 (pad).
kernel void db_homography(device const Box4*  boxes    [[buffer(0)]],
                          device const uint*  boxCount [[buffer(1)]],
                          constant float2&    scale    [[buffer(2)]],   // orig/det
                          constant uint2&     recHW    [[buffer(3)]],   // (H=48, W=320)
                          constant float&     margin   [[buffer(4)]],
                          device float*       H        [[buffer(5)]],   // 9/box
                          device int*         cw       [[buffer(6)]],
                          uint gid [[thread_position_in_grid]])
{
    if(gid >= boxCount[0]){ cw[gid]=0; return; }
    Box4 b=boxes[gid];
    float x0=(float(b.xmin)-margin)*scale.x, y0=(float(b.ymin)-margin)*scale.y;
    float x1=(float(b.xmax)+margin)*scale.x, y1=(float(b.ymax)+margin)*scale.y;
    float bw=max(x1-x0,1.0f), bh=max(y1-y0,1.0f);
    int W=int(clamp(bw*float(recHW.x)/bh, 8.0f, float(recHW.y)));
    cw[gid]=W;
    float sx=bw/float(W), sy=bh/float(recHW.x);
    uint o=gid*9u;
    H[o+0]=sx; H[o+1]=0;  H[o+2]=x0;
    H[o+3]=0;  H[o+4]=sy; H[o+5]=y0;
    H[o+6]=0;  H[o+7]=0;  H[o+8]=1;
}

// ---- Full NVIDIA-parity GPU DB-post: perimeter -> expand -> JFA unclip -> PCA
//      oriented rect. Per-component arrays are indexed by ROOT label (l-1); the
//      JFA key packs d2<<19 | root (root < 2^19), so no compaction is needed.

inline void atomic_min_f(device atomic_uint* a, float v){
    uint old=atomic_load_explicit(a,memory_order_relaxed);
    while(as_type<float>(old) > v){ if(atomic_compare_exchange_weak_explicit(a,&old,as_type<uint>(v),memory_order_relaxed,memory_order_relaxed)) break; }
}
inline void atomic_max_f(device atomic_uint* a, float v){
    uint old=atomic_load_explicit(a,memory_order_relaxed);
    while(as_type<float>(old) < v){ if(atomic_compare_exchange_weak_explicit(a,&old,as_type<uint>(v),memory_order_relaxed,memory_order_relaxed)) break; }
}

// Per-component crack perimeter (exposed 4-edges) ~ cv2.arcLength.
kernel void db_crack_perim(device const uint* label [[buffer(0)]],
                           constant uint2& dims [[buffer(1)]],
                           device atomic_uint* perim [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]){
    uint W=dims.x,H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x; uint l=label[i]; if(l==0u) return;
    uint c=0;
    if(gid.x==0u   || label[i-1]==0u) c++;
    if(gid.x==W-1u || label[i+1]==0u) c++;
    if(gid.y==0u   || label[i-W]==0u) c++;
    if(gid.y==H-1u || label[i+W]==0u) c++;
    if(c) atomic_fetch_add_explicit(&perim[l-1u], c, memory_order_relaxed);
}

// Per-component unclip distance e = count*ratio/perim, clamp[2,24]; 0 if the
// component fails the score (mean prob) or min-count filter -> dropped by JFA.
kernel void db_expand(device const uint* cnt   [[buffer(0)]],
                      device const uint* psum  [[buffer(1)]],
                      device const uint* perim [[buffer(2)]],
                      constant uint&  N        [[buffer(3)]],
                      constant float& ratio    [[buffer(4)]],
                      constant float& boxThresh [[buffer(5)]],
                      device float*   expand   [[buffer(6)]],
                      uint gid [[thread_position_in_grid]]){
    if(gid>=N) return;
    uint n=cnt[gid]; expand[gid]=0.0f;
    if(n<3u) return;
    float score=float(psum[gid])/1024.0f/float(n);
    if(score<boxThresh) return;
    uint p=perim[gid]; if(p==0u) return;
    float e=float(n)*ratio/float(p);
    expand[gid]=clamp(e, 2.0f, 24.0f);
}

// Boundary-disc JFA unclip: each fg boundary pixel stamps a disc of radius
// ceil(e) of its component, atomicMin on packed key (d2<<19 | root).
kernel void db_jfa_scatter(device const uint*  label  [[buffer(0)]],
                           device const float* expand [[buffer(1)]],
                           constant uint2& dims [[buffer(2)]],
                           device atomic_uint* best [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]){
    uint W=dims.x,H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x; uint l=label[i]; if(l==0u) return;
    uint root=l-1u;
    // boundary only
    bool bnd = (gid.x==0u||label[i-1]==0u)||(gid.x==W-1u||label[i+1]==0u)||
               (gid.y==0u||label[i-W]==0u)||(gid.y==H-1u||label[i+W]==0u);
    if(!bnd) return;
    float e=expand[root]; if(e<=0.0f) return;
    int r=int(ceil(e)); float e2=e*e;
    for(int dy=-r;dy<=r;dy++){ int ny=int(gid.y)+dy; if(ny<0||ny>=int(H))continue;
      for(int dx=-r;dx<=r;dx++){ int nx=int(gid.x)+dx; if(nx<0||nx>=int(W))continue;
        int d2=dx*dx+dy*dy; if(float(d2)>e2) continue;
        uint key=(uint(d2)<<19)|(root & 0x7FFFFu);
        atomic_fetch_min_explicit(&best[uint(ny)*W+uint(nx)], key, memory_order_relaxed);
      } }
}

// Resolve winner keys -> expanded label (root+1, 0=bg). Also stamp the seed's
// own pixels (they belong to their component even if no boundary reached them).
kernel void db_jfa_resolve(device const uint* label [[buffer(0)]],
                           device const uint* best  [[buffer(1)]],
                           constant uint2& dims [[buffer(2)]],
                           constant uint& INIT [[buffer(3)]],
                           device uint* expanded [[buffer(4)]],
                           uint2 gid [[thread_position_in_grid]]){
    uint W=dims.x,H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x;
    uint l=label[i]; if(l!=0u){ expanded[i]=l; return; }  // original fg keeps its root+... (l=root+1)
    uint k=best[i];
    expanded[i] = (k==INIT) ? 0u : ((k & 0x7FFFFu)+1u);
}

// Moments over expanded labels (64-bit): n, Sx, Sy, Sxx, Syy, Sxy per root.
kernel void db_moments(device const uint* expanded [[buffer(0)]],
                       constant uint2& dims [[buffer(1)]],
                       device atomic_uint* mom [[buffer(2)]],
                       uint2 gid [[thread_position_in_grid]]){
    uint W=dims.x,H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x; uint e=expanded[i]; if(e==0u) return;
    uint r=e-1u; uint x=gid.x, y=gid.y;
    device atomic_uint* m=mom+(size_t)r*6;
    atomic_fetch_add_explicit(&m[0],1u,memory_order_relaxed);
    atomic_fetch_add_explicit(&m[1],x,memory_order_relaxed);
    atomic_fetch_add_explicit(&m[2],y,memory_order_relaxed);
    atomic_fetch_add_explicit(&m[3],x*x,memory_order_relaxed);
    atomic_fetch_add_explicit(&m[4],y*y,memory_order_relaxed);
    atomic_fetch_add_explicit(&m[5],x*y,memory_order_relaxed);
}

// PCA axis per component + seed projection extents.
kernel void db_axis(device const uint* mom [[buffer(0)]],
                    constant uint& N [[buffer(1)]],
                    device float* orient [[buffer(2)]],   // per root: cos,sin,umin,umax,vmin,vmax
                    uint gid [[thread_position_in_grid]]){
    if(gid>=N) return; device const uint* m=mom+(size_t)gid*6; device float* o=orient+(size_t)gid*6;
    uint n=m[0];
    // seed extents so db_project's atomics work (umin=+inf as float bits)
    o[2]=as_type<float>(0x7f7fffffu); o[3]=as_type<float>(0xff7fffffu); o[4]=o[2]; o[5]=o[3];
    if(n<3u){ o[0]=1.0f; o[1]=0.0f; return; }
    float dn=float(n);
    float mx=float(m[1])/dn, my=float(m[2])/dn;
    float cxx=float(m[3])/dn-mx*mx, cyy=float(m[4])/dn-my*my, cxy=float(m[5])/dn-mx*my;
    float trace=cxx+cyy, aniso=sqrt((cxx-cyy)*(cxx-cyy)+4.0f*cxy*cxy);
    if(trace<=0.0f || aniso<0.05f*trace){ o[0]=1.0f; o[1]=0.0f; return; }
    float th=0.5f*atan2(2.0f*cxy, cxx-cyy);
    o[0]=cos(th); o[1]=sin(th);
}

// Project pixels onto the oriented basis; track min/max along each axis.
kernel void db_project(device const uint* expanded [[buffer(0)]],
                       constant uint2& dims [[buffer(1)]],
                       device const uint* mom [[buffer(2)]],
                       device atomic_uint* orient [[buffer(3)]],  // reinterpret o[2..5] as atomic_uint
                       uint2 gid [[thread_position_in_grid]]){
    uint W=dims.x,H=dims.y; if(gid.x>=W||gid.y>=H) return;
    uint i=gid.y*W+gid.x; uint e=expanded[i]; if(e==0u) return; uint r=e-1u;
    if(mom[(size_t)r*6]<3u) return;
    device atomic_uint* o=orient+(size_t)r*6;
    float c=as_type<float>(atomic_load_explicit(&o[0],memory_order_relaxed));
    float s=as_type<float>(atomic_load_explicit(&o[1],memory_order_relaxed));
    float u=float(gid.x)*c+float(gid.y)*s, v=-float(gid.x)*s+float(gid.y)*c;
    atomic_min_f(&o[2],u); atomic_max_f(&o[3],u); atomic_min_f(&o[4],v); atomic_max_f(&o[5],v);
}

// Compact surviving components AND emit the warp homography for each — from the
// ORIENTED quad (rotated crop, det->original), aspect-preserving content width.
// One dispatch over all roots; atomic-append to the dense H/cw list + count.
kernel void db_emit_quads(device const uint*  mom    [[buffer(0)]],
                          device const float* orient [[buffer(1)]],
                          device const float* expand [[buffer(2)]],
                          constant uint&   N       [[buffer(3)]],
                          constant float2& scale   [[buffer(4)]],   // orig/det
                          constant uint2&  recHW   [[buffer(5)]],   // (H=48, W=320)
                          constant uint&   maxBox  [[buffer(6)]],
                          device float*    Hout    [[buffer(7)]],   // 9/box
                          device int*      cwOut   [[buffer(8)]],
                          device atomic_uint* boxCount [[buffer(9)]],
                          uint gid [[thread_position_in_grid]]){
    if(gid>=N) return;
    if(mom[gid*6]<3u || expand[gid]<=0.0f) return;
    float c=orient[gid*6], s=orient[gid*6+1];
    float umin=orient[gid*6+2], umax=orient[gid*6+3], vmin=orient[gid*6+4], vmax=orient[gid*6+5];
    float2 tl=float2((umin*c - vmin*s)*scale.x, (umin*s + vmin*c)*scale.y);
    float2 tr=float2((umax*c - vmin*s)*scale.x, (umax*s + vmin*c)*scale.y);
    float2 bl=float2((umin*c - vmax*s)*scale.x, (umin*s + vmax*c)*scale.y);
    float w=length(tr-tl), h=length(bl-tl);
    if(min(w,h) < 3.0f) return;
    int Wc=int(clamp(w, 8.0f, float(recHW.y)));
    uint idx=atomic_fetch_add_explicit(boxCount,1u,memory_order_relaxed); if(idx>=maxBox) return;
    cwOut[idx]=Wc;
    float2 ax=(tr-tl)/float(Wc), ay=(bl-tl)/float(recHW.x);
    uint o=idx*9u;
    Hout[o+0]=ax.x; Hout[o+1]=ay.x; Hout[o+2]=tl.x;
    Hout[o+3]=ax.y; Hout[o+4]=ay.y; Hout[o+5]=tl.y;
    Hout[o+6]=0.0f; Hout[o+7]=0.0f; Hout[o+8]=1.0f;
}
