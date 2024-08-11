import setuptools
from tfpy import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tfpy",
    version=__version__,
    description="Direct Construction of Stellarator Shapes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    url="",
    author="Ke Liu",
    author_email="lk2020@mail.ustc.edu.cn",
    license="GNU 3.0",
    packages=setuptools.find_packages(),
)

