from setuptools import setup, find_packages
import os

setup(
    name="deepbach_pytorch",
    version="0.3.6",
    author="Gaetan Hadjeres, JinruGuan",
    author_email='grudan930809@outlook.com', 
    url='https://github.com/Ghadjeres/DeepBach',
    description="DeepBach implementation for harmonization",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={
        "deepbach_pytorch": [
            "*.sh",
            "Dockerfile"
        ],
    },
    install_requires=[
        "torch>=2.0.0",
        "music21>=8.1.0",
        "tqdm",
        "numpy>=1.23.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    zip_safe=False, 
)