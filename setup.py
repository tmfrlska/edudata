from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("VERSION", "r") as f:
    version = f.read().strip()

install_requires = [
    'numpy>=1.17.3',
    'pandas>=0.25.3',
    'scipy>=1.3.1',
    'scikit-learn>=0.22',
    # 'table_evaluator>=1.4.2',  # 의존성 충돌로 인해 주석처리
    'seaborn>=0.11.2'
    ]

setup(
    name="edudata4ai",
    version=version,
    author="Seulki.Kim",
    author_email="tmfrlska85@gmail.com",
    description="Datasets and Synthetic datasets generator for education",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tmfrlska/edudata",
    install_requires=install_requires,   
    packages=find_packages(),
    # package_data={'': ['datasets/*.csv', 'datasets/*.json']},
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
