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
