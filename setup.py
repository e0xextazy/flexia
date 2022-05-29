import os
from setuptools import setup, find_packages


NAME = "torch-components"
DESCRIPTION = "Torch Components is an open-source high-level API above the PyTorch framework. It provides functionality, which can be easily inserted into any PyTorch training and validation scripts."
VERSION = "1.0.3"
AUTHOR = "Vadim Irtlach"
AUTHOR_EMAIL = "vadimirtlach@gmail.com"
URL = "https://github.com/vad13irt/torch-components"
REQUIRED = ["pytimeparse", "numpy", "torch"]
PACKAGES = find_packages("torch_components/*")

DIRECTORY = os.getcwd()
README_PATH = os.path.join(DIRECTORY, "README.md")

with open(README_PATH, "r", encoding="utf-8") as readme:
    README = str(readme.read())

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=PACKAGES,
    include_package_data=True,
    install_requires=REQUIRED,
)