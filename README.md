# info

Small script that shows information about the GPU and CUDA 
Requirements https://github.com/anderskm/gputil
Optional requirements torch, tensorflow

# pip
## required
```
pip install GPUtil
```
## optional
```
pip install tensorflow-gpu==1.15.2
pip install torch==1.7.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

```

# run
```
/usr/local/cuda/bin/nvcc --version
python cuda_info/test.py
```

# out example 
```
ID = 0
Name = Tesla T4
UUID = GPU-eaeadb05-fb7e-c96d-4d36-c067c89286e1
GPU util. = 0%
Memory util. = 85%
Memory total = 15109MB
Memory used = 12898MB
Memory free = 2211MB
torch.cuda.is_available = True
Torch version = 1.7.1
tensorflow.test.is_gpu_available = True
TensorFlow version = 1.15.2
GPU devicename = /device:GPU:0
GPU list = PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
GPU desc = device: 0, name: Tesla T4, pci bus id: 0000:65:00.0, compute capability: 7.5
```