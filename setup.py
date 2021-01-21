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
        'multianndata',
        'numpy',
        'pandas',
        'scipy',
        'argparse',
        'matplotlib',
        'scanpy',
        ],
)
