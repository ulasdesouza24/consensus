from setuptools import setup, find_packages

setup(
    name='consensus-fs',
    version='0.1.1',
    description='Ensemble Feature Selection Library',
    author='Ulaş Taylan Met',
    author_email='umet9711@gmail.com',
    packages=find_packages(),
    install_requires=[
        'scikit-learn>=1.0.0',
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'shap>=0.40.0',
        'lofo-importance>=0.3.0',
        'joblib>=1.0.0',
        'seaborn>=0.11.0',
        'matplotlib>=3.3.0'
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
