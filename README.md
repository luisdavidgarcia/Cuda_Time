# Cuda_Time
A repo dedicated to learning how to program with Cuda

I am following the tutorial found here: [Cuda Programming](https://www.youtube.com/watch?v=86FAWCzIe_4)

## Nvidia Toolkit

Go directly to the Nvidia website to download. Then to ensure you've installed 
correctly do:

```sh
nvcc --version
nvidia-smi
```

If both of these commands work then you are doing good.

NOTE: It is critical to learn C/C++ because this is the software that you 
will be writing your CUDA code in.

## Profiling

For profiling your kernels you can use these CLI commands:

```sh
nvcc -o <object_file> <program>.cu -lnvToolsExt
nsys profile --stats=true ./<object_file>
```

However, to just visualize GPU performance and memory use you can use tools 
like these:

- `nvitop`
- `nvidia-smi` or `watch -n 0.1 nvidia-smi`

If you prefer a GUI though you can use `nsys` and `ncu` 

## Cuda API

To learn more about Cuda and how to utilize it's prebuilt kernels use the 
CUDA API and read the docs: [CUDA API Docs](https://docs.nvidia.com/cuda/)

### cuBLAS
 
 Nivida CUDA Basic Linear Algebra Subprograms is a solid GPU-accelerated library
 for accelerating AI and HPT, but be aware of how you shape you matrix with it
 [link](https://stackoverflow.com/questions/56043539/cublassgemm-row-major-multiplication)

There are two other varations of cuBLAS:

- Lt: The lightweight version so expect some accuracy loss
- Xt: Combines CPU and GPU for larger matrices, but much slower performance

To compile the code you often must use this flag with the `nvcc` compiler:
- If using plain cuBlas or Xt library: `-lcublas`
- If using the Lt version: `-lcublas -lcublasLt`
