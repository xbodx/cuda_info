from info import gpu_info

infos = gpu_info()
for info in infos:
    for inf in info:
        print(inf)
