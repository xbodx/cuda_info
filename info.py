import logging
import os
import platform
import sys
from distutils import spawn
from subprocess import PIPE, Popen

from GPUtil import getGPUs
from GPUtil.GPUtil import safeFloatCast, GPU

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


def getGPUs():
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi
        # could not be found from the environment path,
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
    else:
        nvidia_smi = "nvidia-smi"

    # Get ID, processing and memory utilization for all GPUs
    try:
        p = Popen([nvidia_smi,
                   "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
                   "--format=csv,noheader,nounits"], stdout=PIPE)
        stdout, stderror = p.communicate()
    except Exception as e:
        print(e)
        return []
    output = stdout.decode('UTF-8')
    # output = output[2:-1] # Remove b' and ' from string added by python
    # print(output)
    ## Parse output
    # Split on line break
    lines = output.split(os.linesep)
    # print(lines)
    numDevices = len(lines) - 1
    GPUs = []
    for g in range(numDevices):
        line = lines[g]
        # print(line)
        vals = line.split(', ')
        # print(vals)
        for i in range(12):
            # print(vals[i])
            if (i == 0):
                deviceIds = int(vals[i])
            elif (i == 1):
                uuid = vals[i]
            elif (i == 2):
                gpuUtil = safeFloatCast(vals[i]) / 100
            elif (i == 3):
                memTotal = safeFloatCast(vals[i])
            elif (i == 4):
                memUsed = safeFloatCast(vals[i])
            elif (i == 5):
                memFree = safeFloatCast(vals[i])
            elif (i == 6):
                driver = vals[i]
            elif (i == 7):
                gpu_name = vals[i]
            elif (i == 8):
                serial = vals[i]
            elif (i == 9):
                display_active = vals[i]
            elif (i == 10):
                display_mode = vals[i]
            elif (i == 11):
                temp_gpu = safeFloatCast(vals[i]);
        GPUs.append(GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode,
                        display_active, temp_gpu))
    return GPUs  # (deviceIds, gpuUtil, memUtil)

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

