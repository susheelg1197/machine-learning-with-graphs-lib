from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-16') as f:
    required = f.read().splitlines()


setup(
    name='machine_learning_with_graph',
    version='0.0.2',
    author='Susheel Gounder and Parikshit Urs',
    author_email='susheelg1107@gmail.com',
    description='A comprehensive package for graph-based machine learning algorithms.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/machine-learning-with-graph',
    license='MIT',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='graph neural networks machine learning GNN GCN GAT',
    python_requires='>=3.7',
    project_urls={
        'Bug Reports': 'https://github.com/susheelg1197/machine-learning-with-graph/issues',
        'Source': 'https://github.com/susheelg1197/machine-learning-with-graph',
    },
)
