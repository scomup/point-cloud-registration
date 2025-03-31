from setuptools import setup, find_packages

setup(
    name='point-cloud-registration',
    version='1.0.2',
    author="Liu Yang",
    description="A pure python point cloud registration library.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/scomup/point-cloud-registration",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pykdtree',
    ],
)