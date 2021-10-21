import setuptools

__version__ = '0.0.2'

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="esntorch",
    version="0.0.2",
    description="Python library: Echo state Networks for NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Playtika Ltd.",
    author_email="",
    license="MIT License",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.7',
)
