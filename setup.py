from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
from torch.utils import cpp_extension

setup(name='dag_torch',
      ext_modules=[
          cpp_extension.CppExtension('dag.tmpi_torch',
                                     ['dag/tmpi_torch.cpp'])
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
