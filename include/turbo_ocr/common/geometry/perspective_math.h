#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace turbo_ocr {

/// Compute inverse perspective transform: maps dst quad to src quad.
/// dst_pts[8] = {x0,y0, x1,y1, x2,y2, x3,y3} in output space
/// src_pts[8] = {x0,y0, x1,y1, x2,y2, x3,y3} in source image space
/// M_inv[9] = output 3x3 matrix
///
/// This is a pure-math function with no CUDA dependency, extracted from
/// kernels.h so it can be used in CPU-only code and unit tests.
inline void compute_perspective_inv(
    const float* dst_pts, const float* src_pts,
    float* M_inv) {
  double x0=dst_pts[0],y0=dst_pts[1],x1=dst_pts[2],y1=dst_pts[3];
  double x2=dst_pts[4],y2=dst_pts[5],x3=dst_pts[6],y3=dst_pts[7];
  double u0=src_pts[0],v0=src_pts[1],u1=src_pts[2],v1=src_pts[3];
  double u2=src_pts[4],v2=src_pts[5],u3=src_pts[6],v3=src_pts[7];

  double A[8][8] = {
    {x0,y0,1, 0,0,0, -x0*u0,-y0*u0},
    {x1,y1,1, 0,0,0, -x1*u1,-y1*u1},
    {x2,y2,1, 0,0,0, -x2*u2,-y2*u2},
    {x3,y3,1, 0,0,0, -x3*u3,-y3*u3},
    {0,0,0, x0,y0,1, -x0*v0,-y0*v0},
    {0,0,0, x1,y1,1, -x1*v1,-y1*v1},
    {0,0,0, x2,y2,1, -x2*v2,-y2*v2},
    {0,0,0, x3,y3,1, -x3*v3,-y3*v3}
  };
  double b[8] = {u0,u1,u2,u3,v0,v1,v2,v3};

  for (int col=0;col<8;col++) {
    int piv=col; double mx=std::abs(A[col][col]);
    for (int r=col+1;r<8;r++) if (std::abs(A[r][col])>mx) {mx=std::abs(A[r][col]);piv=r;}
    if (piv!=col) {std::swap(b[col],b[piv]); for(int k=0;k<8;k++) std::swap(A[col][k],A[piv][k]);}
    double d=A[col][col]; if(std::abs(d)<1e-12) d=(d<0)?-1e-12:1e-12;
    for(int k=col;k<8;k++) { A[col][k]/=d; } b[col]/=d;
    for(int r=0;r<8;r++) {if(r==col)continue; double f=A[r][col]; for(int k=col;k<8;k++) A[r][k]-=f*A[col][k]; b[r]-=f*b[col];}
  }
  M_inv[0]=(float)b[0];M_inv[1]=(float)b[1];M_inv[2]=(float)b[2];
  M_inv[3]=(float)b[3];M_inv[4]=(float)b[4];M_inv[5]=(float)b[5];
  M_inv[6]=(float)b[6];M_inv[7]=(float)b[7];M_inv[8]=1.0f;
}

} // namespace turbo_ocr
