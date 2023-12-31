# Deep_Orga
Deep_Orga model maintains the single-stage and concise features of the YOLOX model while surpassing the performance of the classical models.

![Total pipline of the Deep_Orga for organoid detection](https://github.com/sibet-lb/Deep_Orga/blob/main/Graphical%20Abstract.png)

## Dataset
The dataset utilized in this paper was obtained from the literature(T. Kassis, V. Hernandez-Gordillo, R. Langer, L.G. Griffith, 1002-07. OrgaQuant: Human Intestinal Organoid Localization and Quantification Using Deep Convolutional Neural Networks, Sci. Rep. 9 (2019) 1–7. https://doi.org/10.1038/s41598-019-48874-y.) and consists of bright field microscope images of organoid cultures. These organoids were derived from patient duodenal biopsy tissues and were cultured following ethical review and with patient consent. To label the dataset, a crowdsourcing platform was employed, and manual annotation was performed. The dataset comprises 1750 images and contains a total of 14242 organoids.

## Toolbox
MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We use MMDetection in our research.
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}

## Getting Started
##System environment:
    sys.platform: win32
    Python: 3.11.3 | packaged by Anaconda, Inc. | (main, May 15 2023, 15:41:31) [MSC v.1916 64 bit (AMD64)]
    CUDA available: True
    numpy_random_seed: 1494133411
    GPU 0: NVIDIA GeForce RTX 2060
    CUDA_HOME: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
    NVCC: Cuda compilation tools, release 11.8, V11.8.89
    MSVC: 用于 x64 的 Microsoft (R) C/C++ 优化编译器 19.34.31935 版
    GCC: n/a
    PyTorch: 2.0.1
    PyTorch compiling details: PyTorch built with:
  - C++ Version: 199711
  - MSVC 193431937
  - Intel(R) Math Kernel Library Version 2020.0.2 Product Build 20200624 for Intel(R) 64 architecture applications
  - Intel(R) MKL-DNN v2.7.3 (Git Hash 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
  - OpenMP 2019
  - LAPACK is enabled (usually provided by MKL)
  - CPU capability usage: AVX2
  - CUDA Runtime 11.8
  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_37,code=compute_37
  - CuDNN 8.7
  - Magma 2.5.4
  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.8, CUDNN_VERSION=8.7.0, CXX_COMPILER=C:/cb/pytorch_1000000000000/work/tmp_bin/sccache-cl.exe, CXX_FLAGS=/DWIN32 /D_WINDOWS /GR /EHsc /w /bigobj /FS -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=OFF, TORCH_VERSION=2.0.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=OFF, USE_OPENMP=ON, USE_ROCM=OFF, 

    TorchVision: 0.15.2
    OpenCV: 4.7.0
    MMEngine: 0.7.4
