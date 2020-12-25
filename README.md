# info

Small script that shows information about the GPU and CUDA

# pyenv

```
pyenv install --list
pyenv install 3.7.9
pyenv versions
pyenv local 3.7.9
```

# venv

in windows
```
python -m venv .cuda_info_env
.\.cuda_info_env\Scripts\activate.bat 
python -m pip install --upgrade pip
```

in linux
```
python -m venv .cuda_info_env
source ./.cuda_info_env/bin/activate
python -m pip install --upgrade pip
```

# pip

required
```
pip install GPUtil
```

optional
```
pip install tensorflow-gpu==1.15.2
pip install torch
```

# run

```
python main.py
```


# example 

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