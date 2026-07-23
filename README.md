## Introduction
This repo tracks my from-scratch attempt to optimize a CUDA SGEMM kernel toward cuBLAS-like performance, following the structure of Simon Boehm's article ["How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"](https://siboehm.com/articles/22/CUDA-MMM). My approach: read a section of the article, then implement that kernel myself without looking at Simon's code, then compare notes.

One deliberate deviation: I started at Kernel 2 rather than Kernel 1. My "naive" kernel already maps threads to coalesce global memory reads, which is closer to Boehm's Kernel 2 than his intentionally-uncoalesced Kernel 1 — so I used it as my baseline rather than re-deriving an already-solved problem.

All kernels are benchmarked on an NVIDIA T4 (Turing) via Lightning.AI, profiled with Nsight Compute. I'm learning CUDA and GPU architecture as I go — if you spot something I got wrong, or have resources you think would help, I'd genuinely appreciate an email at [jribaneze@gmail.com](mailto:jribaneze@gmail.com).).
