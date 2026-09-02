// FP16 텐서코어 피크와 **FP32 누산 반감 여부**를 mma.sync 로 직접 잰다.
//
// 왜 cuBLAS 로 재지 않는가
// -----------------------
// cuBLAS 실측에는 커널 품질(타일링/스케줄링/메모리)이 섞인다. 우리가 알고
// 싶은 것은 하드웨어의 FMA/clk/SM 이고, 그것은 **메모리 접근이 없는 순수
// mma.sync 루프**에서만 깨끗하게 나온다. cuBLAS 는 교차검증에만 쓴다.
//
// 무엇을 내는가
// -------------
//   1) clock64() 기반 FMA/clk/SM   ★ 클럭에 의존하지 않는 값
//   2) 벽시계 기반 TFLOP/s          (그 시점 클럭에서의 실효 처리량)
//   3) 두 값에서 역산한 SM 클럭     ★ 두 모드의 역산 클럭이 일치해야
//                                    측정이 건전한 것이다
//
// FP32 누산(f32.f16.f16.f32)과 FP16 누산(f16.f16.f16.f16)을 같은 절차로
// 재서 비율을 본다. 비율이 ~2 면 FP32 누산 반감이 있는 SKU 다
// (GeForce Ada/Blackwell). ~1 이면 없다 (A6000 같은 프로 SKU).
//
//   nvcc -arch=sm_90a -O3 tools/peak_mma_probe.cu -o peak_mma_probe && ./peak_mma_probe
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

#ifndef KT_ITERS
#define KT_ITERS 8192          // 바깥 루프. 안쪽에 독립 mma 8 개.
#endif
#define KT_CHAINS 8            // 의존성 지연을 덮기 위한 독립 누산기 수

__global__ void probe_f32(int iters, float *out, long long *cycles) {
  uint32_t a0=0x3c003c00u,a1=0x3c003c00u,a2=0x3c003c00u,a3=0x3c003c00u;
  uint32_t b0=0x3c003c00u,b1=0x3c003c00u;
  float d[KT_CHAINS][4];
#pragma unroll
  for (int c=0;c<KT_CHAINS;++c) for (int i=0;i<4;++i) d[c][i]=0.f;
  __syncthreads();
  long long t0 = clock64();
  for (int it=0; it<iters; ++it) {
#pragma unroll
    for (int c=0;c<KT_CHAINS;++c) {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[c][0]),"+f"(d[c][1]),"+f"(d[c][2]),"+f"(d[c][3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
  }
  long long t1 = clock64();
  float s=0.f;
#pragma unroll
  for (int c=0;c<KT_CHAINS;++c) for (int i=0;i<4;++i) s+=d[c][i];
  if (threadIdx.x==0) cycles[blockIdx.x] = t1-t0;
  if (s == 12345.678f) out[blockIdx.x] = s;   // 최적화 제거 방지
}

__global__ void probe_f16(int iters, float *out, long long *cycles) {
  uint32_t a0=0x3c003c00u,a1=0x3c003c00u,a2=0x3c003c00u,a3=0x3c003c00u;
  uint32_t b0=0x3c003c00u,b1=0x3c003c00u;
  uint32_t d[KT_CHAINS][2];
#pragma unroll
  for (int c=0;c<KT_CHAINS;++c) { d[c][0]=0u; d[c][1]=0u; }
  __syncthreads();
  long long t0 = clock64();
  for (int it=0; it<iters; ++it) {
#pragma unroll
    for (int c=0;c<KT_CHAINS;++c) {
      asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(d[c][0]),"+r"(d[c][1])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
  }
  long long t1 = clock64();
  uint32_t s=0u;
#pragma unroll
  for (int c=0;c<KT_CHAINS;++c) { s+=d[c][0]; s+=d[c][1]; }
  if (threadIdx.x==0) cycles[blockIdx.x] = t1-t0;
  if (s == 0xdeadbeefu) out[blockIdx.x] = 1.f;
}

struct R { double tflops, fma_per_clk_sm, implied_mhz, ms; long long cyc; };

template <class K>
R run(K kern, const char *tag, int sm, int warps, int iters, bool quiet=false) {
  int threads = warps*32, blocks = sm;           // SM 당 정확히 블록 하나
  float *out; long long *cyc;
  cudaMalloc(&out, blocks*sizeof(float));
  cudaMalloc(&cyc, blocks*sizeof(long long));
  kern<<<blocks,threads>>>(iters/8, out, cyc);   // 워밍업
  cudaDeviceSynchronize();
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  kern<<<blocks,threads>>>(iters, out, cyc);
  cudaEventRecord(e1); cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) { printf("%s: %s\n", tag, cudaGetErrorString(err)); return {}; }
  float ms=0; cudaEventElapsedTime(&ms,e0,e1);
  long long *h = new long long[blocks];
  cudaMemcpy(h,cyc,blocks*sizeof(long long),cudaMemcpyDeviceToHost);
  long long mx=0; double sum=0;
  for (int i=0;i<blocks;++i){ sum+=h[i]; if(h[i]>mx) mx=h[i]; }
  double cyc_avg = sum/blocks;

  // MAC 수: mma m16n8k16 하나가 워프당 16*8*16 = 2048 MAC
  double macs_per_sm = (double)warps * iters * KT_CHAINS * 2048.0;
  double macs_total  = macs_per_sm * sm;
  double tflops = macs_total*2.0/(ms*1e-3)/1e12;
  double fma_clk_sm = macs_per_sm / cyc_avg;
  double implied_mhz = (macs_per_sm/ (ms*1e-3)) / fma_clk_sm / 1e6;
  delete[] h; cudaFree(out); cudaFree(cyc);
  cudaEventDestroy(e0); cudaEventDestroy(e1);
  if (!quiet)
    printf("  %-12s warps=%-2d  %8.2f TFLOP/s   FMA/clk/SM %7.1f   "
           "역산클럭 %6.0f MHz   cycles %.3e (편차 %.1f%%)\n",
           tag, warps, tflops, fma_clk_sm, implied_mhz, cyc_avg,
           100.0*(mx-cyc_avg)/cyc_avg);
  return {tflops, fma_clk_sm, implied_mhz, ms, (long long)cyc_avg};
}

int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
  printf("%s  sm_%d%d  SM=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
  printf("mma.sync.aligned.m16n8k16 (메모리 접근 없음, 독립 누산기 %d개)\n\n", KT_CHAINS);
  int sm = p.multiProcessorCount;
  R best32{}, best16{};
  for (int w : {4, 8, 12, 16, 24, 32}) {
    R a = run(probe_f32, "FP32 누산", sm, w, KT_ITERS);
    R b = run(probe_f16, "FP16 누산", sm, w, KT_ITERS);
    if (a.tflops > best32.tflops) best32 = a;
    if (b.tflops > best16.tflops) best16 = b;
    printf("      비율(FP16누산/FP32누산) = %.3f\n\n", b.tflops/a.tflops);
  }
  printf("최대: FP32 누산 %.2f TFLOP/s (FMA/clk/SM %.1f)  |  "
         "FP16 누산 %.2f TFLOP/s (FMA/clk/SM %.1f)  |  비율 %.3f\n",
         best32.tflops, best32.fma_per_clk_sm, best16.tflops,
         best16.fma_per_clk_sm, best16.tflops/best32.tflops);
  printf("역산 클럭: FP32 %.0f MHz / FP16 %.0f MHz  (일치해야 측정이 건전하다)\n",
         best32.implied_mhz, best16.implied_mhz);
  return 0;
}
