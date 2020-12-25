import logging
import sys

from GPUtil import getGPUs

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
    GPUs = getGPUs()
    attrList = [
        [{'attr': 'id', 'name': 'ID'},
         {'attr': 'name', 'name': 'Name'},
         {'attr': 'uuid', 'name': 'UUID'}],
        [{'attr': 'load', 'name': 'GPU util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0},
         {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '%', 'transform': lambda x: x * 100,
          'precision': 0}],
        [{'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
         {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0},
         {'attr': 'memoryFree', 'name': 'Memory free', 'suffix': 'MB', 'precision': 0}]
    ]

    headers = []
    infos = [[]] * len(GPUs)
    for attrGroup in attrList:
        # print(attrGroup)
        for attrDict in attrGroup:
            headers.append(attrDict['name'])

            attrPrecision = '.' + str(attrDict['precision']) if ('precision' in attrDict.keys()) else ''
            attrSuffix = str(attrDict['suffix']) if ('suffix' in attrDict.keys()) else ''
            attrTransform = attrDict['transform'] if ('transform' in attrDict.keys()) else lambda x: x
            for gpu in GPUs:
                attr = getattr(gpu, attrDict['attr'])
                attr = attrTransform(attr)

                if (isinstance(attr, float)):
                    attrStr = ('{0:' + attrPrecision + 'f}').format(attr)
                elif (isinstance(attr, int)):
                    attrStr = ('{0:d}').format(attr)
                elif (isinstance(attr, str)):
                    attrStr = attr;
                elif (sys.version_info[0] == 2):
                    if (isinstance(attr, unicode)):
                        attrStr = attr.encode('ascii', 'ignore')
                else:
                    raise TypeError(
                        'Unhandled object type (' + str(type(attr)) + ') for attribute \'' + attrDict['name'] + '\'')

            for gpuIdx, gpu in enumerate(GPUs):
                attr = getattr(gpu, attrDict['attr'])
                attr = attrTransform(attr)

                if (isinstance(attr, float)):
                    attrStr = ('{0:' + attrPrecision + 'f}').format(attr)
                elif (isinstance(attr, int)):
                    attrStr = '{0:0d}'.format(attr)
                elif (isinstance(attr, str)):
                    attrStr = attr
                elif (sys.version_info[0] == 2):
                    if (isinstance(attr, unicode)):
                        attrStr = attr.encode('ascii', 'ignore')
                else:
                    raise TypeError(
                        'Unhandled object type (' + str(type(attr)) + ') for attribute \'' + attrDict['name'] + '\'')

                attrStr += attrSuffix
                infos[gpuIdx].append(attrStr)

    result = []
    for info in infos:
        for ziped in zip(headers, info):
            result.append(f'{ziped[0]} = {ziped[1]}')
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


def gpu_info():
    result = [get_GPU_info(), get_Torch_info(), get_TensorFlow_info()]
    return result

