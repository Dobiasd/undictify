import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="undictify",
    version="0.11.3",
    author="Tobias Hermann",
    author_email="editgym@gmail.com",
    description="Type-checked function calls at runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/Dobiasd/undictify",
    package_data={"undictify": ["py.typed"]},
    packages=["undictify"],
    python_requires=">=3.8",
    classifiers=(
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
