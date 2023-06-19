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
    'torch==1.10.2',
    'transformers',
    'ipdb',
    'datasets',
    'jax==0.3.21',
    'jaxlib==0.3.20',   # NOTE: use this to train on GPU: pip install jaxlib==0.3.20+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
    'flax==0.6.1',
    'sentencepiece',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.9',
  ],
)
