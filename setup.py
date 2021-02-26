from distutils.core import setup

setup(
      name = 'cuda_info',
      packages = ['cuda_info'],
      version = '0.0.1',
      description = 'cuda_info is a Python module for getting the GPU info from NVIDA GPUs using nvidia-smi via GPUtil and and optionally via torch, tensorflow.',
      author = 'Roman Kasovsky',
      author_email = 'roman@kasovsky.ru',
      url = 'https://github.com/xbodx/cuda_info.git ',
      keywords = ['gpu','utilization','load','memory','available','usage','free','select','nvidia', 'cuda', 'tensorflow', 'torch'],
      classifiers = [],
      license = 'MIT',
      setup_requires=['GPUtil'],
      install_requires=['GPUtil'],
)
