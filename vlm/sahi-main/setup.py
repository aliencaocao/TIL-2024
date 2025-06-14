import io
import os
import re

import setuptools


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, "README.md"), encoding="utf-8") as f:
        return f.read()


def get_requirements():
    with open("requirements.txt", encoding="utf8") as f:
        return f.read().splitlines()


def get_version():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    version_file = os.path.join(current_dir, "sahi", "__init__.py")
    with io.open(version_file, encoding="utf-8") as f:
        return re.search(r'^__version__ = [\'"]([^\'"]*)[\'"]', f.read(), re.M).group(1)


setuptools.setup(
    name="sahi",
    version=get_version(),
    author="OBSS",
    license="MIT",
    description="A vision library for performing sliced inference on large images/small objects",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/obss/sahi",
    packages=setuptools.find_packages(
        exclude=["sahi.models.yolov6", "demo", "docs", "resources", "tests", "scripts"]
    ),
    python_requires=">=3.6",
    install_requires=get_requirements(),
    extras_require={
        "tests": ["mmdet==3.0.0", "pycocotools==2.0.6"],
        "dev": [
            "black==22.3.0",
            "flake8==3.9.2",
            "importlib-metadata>=1.1.0,<4.3;python_version<'3.8'",
            "isort==5.9.2",
            "jupyterlab==3.0.14",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "sahi=sahi.cli:app",
        ],
    },
)
