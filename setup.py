import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pytorch-hebbian",
    version="0.1.1",
    author="Jules Talloen",
    author_email="jules@talloen.eu",
    description="Lightweight framework for Hebbian learning based on PyTorch Ignite",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch==1.6.0',
        'torchvision==0.7.0',
        'tensorboard==2.3.0',
        'pytorch-ignite==0.4.*',
        'matplotlib',
        'numpy',
        'Pillow',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'tqdm',
        'wrapt',
    ],
)
