import resource
import subprocess
import re

def track_resource_consumption(func):
    def wrapper(*args, **kwargs):
        start_time = resource.getrusage(resource.RUSAGE_SELF)
        result = func(*args, **kwargs)
        end_time = resource.getrusage(resource.RUSAGE_SELF)

        # Calculate resource consumption
        cpu_time = end_time.ru_utime - start_time.ru_utime
        memory_usage = end_time.ru_maxrss

        print(f"CPU Time: {cpu_time} seconds")
        print(f"Memory Usage: {memory_usage} KB")

        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            output = result.stdout.strip()

            # Parsing GPU usage information from nvidia-smi output
            pattern = r"(\d+)\s+MiB\s/\s+(\d+)\s+MiB\s+\|\s+(\d+)\%\sDefault"
            match = re.search(pattern, output)
            if match:
                used_memory = int(match.group(1))
                total_memory = int(match.group(2))
                gpu_utilization = int(match.group(3))
                return used_memory, total_memory, gpu_utilization
            else:
                return None

        except FileNotFoundError:
            print("nvidia-smi command not found. Make sure you have NVIDIA drivers installed.")


        return result, used_memory, total_memory, gpu_utilization

    return wrapper

# Example usage:
@track_resource_consumption
def my_function():
    for i in range(1000000):
        a = 15.6845 + 0.000684

my_function()