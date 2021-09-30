import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cna",
    version="0.0.7",
    author="Yakir Reshef, Laurie Rumker",
    author_email="yreshef@broadinstitute.org",
    description="covarying neighborhood analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yakirr/cna",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'multianndata>=0.0.4',
        'anndata>=0.7.1',
        'numpy>=1.18.1',
        'pandas>=1.0.3',
        'scipy>=1.4.1',
        'argparse>=1.1',
        'matplotlib>=3.1.3',
        'scanpy>=1.4.4.post1',
        ],
)
