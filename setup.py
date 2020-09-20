from setuptools import setup, find_packages
from os import path


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='relaxNMR',
    version='0.0.1',
    description='1D Proton NMR relaxometry analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/specmicp/relaxNMR',
    author='Fabien Georget',
    author_email='Fabien.georget@epfl.ch',
    license='BSD',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='NMR analysis relaxometry ',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'test']),
    python_requires='>=3.4, <4',
    install_requires=[
            'numpy',
            'scipy'
            ],
    data_files=[],
    project_urls={
        'Funding': 'https://lmc.epfl.ch/'
    },
)
