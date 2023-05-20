from setuptools import setup, find_packages

setup(
  name = 'babylm_baseline_train',
  package_dir={"": "src"},
  packages=find_packages("src"),
  version = '1.0.0',
  license='MIT',
  description = 'BabyLM Baseline Training',
  long_description_content_type = 'text/markdown',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
  ],
  install_requires=[
    'torch>=1.6',
    'transformers',
    'ipdb',
    'datasets',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)
