from setuptools import setup, find_packages

setup(
    name='vtacML',
    version='0.1.20',
    packages=find_packages(include=['vtacML']),
    install_requires=[
        'numpy==1.26.3',
        'matplotlib==3.8.0',
        'pandas==2.1.4',
        'scikit-learn==1.3.0',
        'seaborn==0.12.2',
        'yellowbrick==1.5',
        'pyyaml==6.0.1',
        'imblearn==0.0',
        'fastparquet==2023.8.0',
        'joblib==1.2.0'
        # List your dependencies here
        # Example: 'numpy', 'pandas', 'scikit-learn',
    ],
    extras_require={
        'dev': [
            'pytest==8.0.1',
            'pylint==3.2.6',
            'black==24.4.2'

        ]

    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            # Define any command-line scripts here
            # Example: 'vtac_classifier=pipeline:main',
        ],
    },
    url='https://github.com/jerbeario/VTAC_ML',  # Replace with your project's URL
    license='MIT',
    author='Jeremy Palmerio',
    author_email='jeremypalmerio05@gmail.com',
    description='A machine learning pipeline to classify objects in VTAC dataset as GRB or not.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',
)
