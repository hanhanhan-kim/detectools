from setuptools import setup, find_packages

with open("README.md","r") as fh:
    long_description = fh.read()

setup(
    name="detectools",
    version="0.0.0",
    author="Han Kim",
    author_email="hankim@caltech.edu",
    description="CLI wrapper for working with Detectron2",
    long_description=long_description,
    url="https://github.com/hanhanhan-kim/detectools",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
    entry_points={
        "console_scripts": ["detectools=detectools.detectools:cli"]
    },
    install_requires=[
        "pyyaml",
        "tqdm",
        "click",
        "opencv-python",
        "detectron2"
    ]

)