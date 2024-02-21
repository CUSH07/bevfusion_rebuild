import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Define the root directory for extension sources
_ext_src_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Define source file patterns for each module
iou3d_sources = glob.glob(os.path.join(_ext_src_root, "iou_nms/src/*.cpp")) + \
                glob.glob(os.path.join(_ext_src_root, "iou_nms/src/*.cu"))

bev_pool_sources = glob.glob(os.path.join(_ext_src_root, "bev_pool/src/*.cpp")) + \
                   glob.glob(os.path.join(_ext_src_root, "bev_pool/src/*.cu"))

requirements = ["torch>=1.4"]

setup(
    name='custom_cuda_extensions',
    version='1.0',
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name='iou3d_nms_cuda',
            sources=iou3d_sources
        ),
        CUDAExtension(
            name='bev_pool_cuda',
            sources=bev_pool_sources
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True
)
#使用from custom_cuda_extensions import iou3d_nms_cuda, bev_pool_cuda调用