import setuptools  # this is the "magic" import
from setuptools import find_packages
from numpy.distutils.core import setup

from sdmapper import __version__


setup(
    name = 'sdmapper',
    version = __version__,

    packages = find_packages(),

    # metadata for upload to PyPI
    author = "Shifan Zuo",
    author_email = "sfzuo@bao.ac.cn",
    description = "Map-making package for single dish astronomical data.",
    license = "GPL v3.0",
    url = "https://github.com/zuoshifan/sdmapper.git",
)