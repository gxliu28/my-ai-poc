from setuptools import setup, find_packages

setup(
    name="my-ai-poc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "datasets",
        "pytest"
    ],
)
