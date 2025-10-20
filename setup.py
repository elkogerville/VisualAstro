from setuptools import setup, find_packages

setup(
    name='VisualAstro',
    version='0.0.1',
    description='Astro data analysis and plotting package',
    url='https://github.com/elkogerville/VisualAstro',
    author='Elko Gerville-Reache',
    author_email='elkogerville@gmail.com',
    license='MIT',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(where="src", exclude=[
        'ANIMATIONS*',
        'DOCUMENTATION*',
        'Initial_Conditions*',
        'Tests*',
        '*.egg-info*',
        'build*',
        'dist*',
    ]),
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={
        'visualastro': ['stylelib/*.mplstyle'],
    },
    install_requires=[
        'astropy',
        'numpy',
        'matplotlib',
        'regions',
        'scipy',
        'spectral_cube',
        'specutils',
        'tqdm',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
