from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import numpy as np


cpp_extension = [
    Extension(
        "utils.trend", 
        sources=["src/trend.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"], 
    ),
    Extension(
        "utils.skew", 
        sources=["src/skew.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"], 
    ),
    Extension(
        "utils.kurtosis", 
        sources=["src/kurtosis.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"], 
    ),
    Extension(
        "utils.lb", 
        sources=["src/lb.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"], 
    ),
    Extension(
        "utils.adf", 
        sources=["src/adftest.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"], 
    ),
    Extension(
        "utils.acf", 
        sources=["src/acf.cpp"],
        include_dirs=[np.get_include()],
        language="c++",
        extra_compile_args=["-std=c++11"], 
    )
]

# setup 函数的配置
setup(
    name="tsds",
    version="0.1.0",
    author="Hui Wang",
    author_email="huiw1128@gmail.com",
    description="Time Series Description",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wanghui5801/tsds",
    packages=setuptools.find_packages(),
    ext_modules=cpp_extension,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.0.0',
        'statsmodels>=0.13.0',
        'matplotlib>=3.0.0',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
