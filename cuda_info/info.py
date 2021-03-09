import logging
import sys

from GPUtil import getGPUs, GPU

installed_torch = False
try:
    import torch

    installed_torch = True
except ImportError:
    logging.warning('torch is not installed')

installed_tensorflow = False
try:
    import tensorflow
    from tensorflow_core.python.client import device_lib

    installed_tensorflow = True
except ImportError:
    logging.warning('tensorflow is not installed')


def to_text(sentence):
    res = []
    if isinstance(sentence, list):
        for phrase in sentence:
            res.append(' '.join(str(phrase).split()))
    elif isinstance(sentence, str) and sentence.find('\n'):
        for phrase in sentence.strip().splitlines():
            res.append(' '.join(str(phrase).split()))
    else:
        res.append(' '.join(str(sentence).split()))
    return '\n'.join(res)

def get_GPU_info() -> str:
    """
    refactoring https://github.com/anderskm/gputil/blob/master/GPUtil/GPUtil.py showUtilization
    replace print to string list
    """
    try:
        infos = getGPUs()
    except Exception as e:
        print(e)
        infos = []
    result = []
    for info in infos:
        gpu: GPU = info
        result.append(f'ID = {gpu.id}')
        result.append(f'Name = {gpu.name}')
        result.append(f'UUID = {gpu.uuid}')
        result.append(f'Serial = {gpu.serial}')
        result.append(f'Load = {gpu.load}')
        result.append(f'Memory Total = {gpu.memoryTotal}')
        result.append(f'Memory Used = {gpu.memoryUsed}')
        result.append(f'Memory Free = {gpu.memoryFree}')
        result.append(f'Memory Util = {gpu.memoryUtil}')
        result.append(f'Driver = {gpu.driver}')
        result.append(f'Temperature = {gpu.temperature}')
        result.append(f'Display Mode = {gpu.display_mode}')
        result.append(f'Display Active = {gpu.display_active}')
    return result


def get_Torch_info() -> str:
    result = []
    if installed_torch:
        result.append(f"torch.cuda.is_available = {'True' if torch.cuda.is_available() else 'False'}")
        result.append(f"Torch version = {torch.__version__}")
        if hasattr(torch, 'cuda_version'):
            result.append(f"Torch cuda_version = {torch.cuda_version}")
    return result


def get_TensorFlow_info() -> str:
    result = []
    if installed_tensorflow:
        result.append(f"tensorflow.test.is_gpu_available = {'True' if tensorflow.test.is_gpu_available() else 'False'}")
        if tensorflow.test.is_gpu_available():
            result.append(f"TensorFlow version = {tensorflow.__version__}")
            result.append(f"GPU devicename = {tensorflow.test.gpu_device_name()}")
            result.append(f"GPU list = {to_text(tensorflow.config.experimental.list_physical_devices('GPU'))}")
            result.append(f"GPU desc = {to_text([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU'])}")
    return result


def get_gpu_info():
    result = [get_GPU_info(), get_Torch_info(), get_TensorFlow_info()]
    return result

