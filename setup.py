import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="undictify",
    version="0.10.0",
    author="Tobias Hermann",
    author_email="editgym@gmail.com",
    description="Type-checked function calls at runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/Dobiasd/undictify",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
