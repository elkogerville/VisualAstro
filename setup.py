from setuptools import setup, find_packages

setup(
    name='Astrophysical_Visualization_System',
    version='0.0.1',
    description='Astro data analysis and plotting package',
    url='https://github.com/elkogerville/Astrophysical_Visualization_System',
    author='Elko Gerville-Reache',
    author_email='elkogerville@gmail.com',
    license='MIT',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=find_packages(exclude=[
        'ANIMATIONS*',
        'DOCUMENTATION*',
        'Initial_Conditions*',
        'Tests*',
	'MSG_Nbody.egg-info*',
        '*.egg-info*',
        'build*',
        'dist*',
    ]),
    package_data={
        'AVS': ['stylelib/*.mplstyle'],
    },
    install_requires=[
        'astropy'
        'numpy',
        'matplotlib',
        'scipy',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
