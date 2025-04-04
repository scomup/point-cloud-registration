from setuptools import setup, find_packages

setup(
    name='point-cloud-registration',
    version='1.0.3',
    author="Liu Yang",
    description="A fast and lightweight point cloud registration library implemented purely in Python.",
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