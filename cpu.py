import psutil
from datetime import datetime
import time
import logging

cpu_filename = "cpu_logs_test-1.txt"
mem_filename = "mem_logs_test-1.txt"


cpu_file = open(cpu_filename, "w")
no_cores = psutil.cpu_count()
cpu_header = "datetime;"+";".join([str(i) for i in range(1,no_cores+1)])
cpu_file.write(cpu_header)
cpu_file.close()

mem_file = open(mem_filename,"w")
mem_header = "datetime;util"
mem_file.write(mem_header)
mem_file.close()


cpu_file = open(cpu_filename, "a")
mem_file = open(mem_filename, "a")

while True:
    now = datetime.now()
    per_cpu = psutil.cpu_percent(percpu=True)
    cpu_file.write(f"\n{now.strftime('%m/%d/%Y %H:%M:%S')};"+";".join([str(usage) for usage in per_cpu]))
    
    mem_util = psutil.virtual_memory().percent
    mem_file.write(f"\n{now.strftime('%m/%d/%Y %H:%M:%S')};{mem_util}")
    
    time.sleep(5)