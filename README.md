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