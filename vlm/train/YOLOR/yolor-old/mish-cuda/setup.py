from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

EXT_SRCS = [
    'csrc/cpu/mish_cpu.cpp', 'csrc/cuda/mish_cuda.cpp',
    'csrc/cuda/mish_kernel.cu'
]
HEADERS = [
    'csrc/cpu/mish_cpu.h', 'csrc/cuda/mish_cuda.h',
    'csrc/cuda/CUDAApplyUtils.cuh'
]

setup(name='mish_cuda',
      version='0.0.3',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      include_package_data=True,
      zip_safe=False,
      install_requires=['torch>=1.2'],
      ext_modules=[
          CUDAExtension('mish_cuda._C',
                        headers=HEADERS,
                        sources=EXT_SRCS,
                        extra_compile_args={
                            'cxx': [],
                            'nvcc': ['--expt-extended-lambda']
                        })
      ],
      cmdclass={'build_ext': BuildExtension})
