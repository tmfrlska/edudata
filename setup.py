from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="edudata4ai",
    version=0.3.1,
    author="Seulki.Kim",
    author_email="tmfrlska85@gmail.com",
    description="Datasets and Synthetic datasets generator for education",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tmfrlska/edudata.git",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
