from pynvml import *

# 初始化NVML库
nvmlInit()


def get_gpu_info(gpu_index):
    handle = nvmlDeviceGetHandleByIndex(gpu_index)

    # 获取GPU名称
    name = nvmlDeviceGetName(handle)
    log_str = f"GPU {gpu_index}: {name}\n"

    # 获取GPU温度
    temperature = nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)
    log_str += f"  Temperature: {temperature} °C\n"

    # 获取GPU功耗
    power_draw = nvmlDeviceGetPowerUsage(handle)
    power_draw_watts = power_draw / 1000  # 转换为瓦特
    log_str += f"  Power Draw: {power_draw_watts:.2f} W\n"

    # 获取GPU利用率
    utilization = nvmlDeviceGetUtilizationRates(handle)
    log_str += f"  Utilization: {utilization.gpu}%\n"

    # 获取GPU显存信息
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    log_str += f"  Total Memory: {mem_info.total / 1024 ** 2:.2f} MB\n"
    log_str += f"  Used Memory: {mem_info.used / 1024 ** 2:.2f} MB\n"
    log_str += f"  Free Memory: {mem_info.free / 1024 ** 2:.2f} MB\n"
    return log_str


# 获取GPU的数量
deviceCount = nvmlDeviceGetCount()
print(f"Number of GPUs: {deviceCount}")
gpu_info = get_gpu_info(0)
print(gpu_info)
# 关闭NVML
nvmlShutdown()
