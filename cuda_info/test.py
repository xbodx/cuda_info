from info import get_gpu_info

infos = get_gpu_info()
for info in infos:
    for inf in info:
        print(inf)
