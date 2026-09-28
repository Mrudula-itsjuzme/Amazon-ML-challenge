import os, time
from datetime import timedelta

HZ = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
btime = 0
with open('/proc/stat') as f:
    for line in f:
        if line.startswith('btime '):
            btime = int(line.split()[1])
            break

print(f"{'PID':<8} {'TIME RUNNING':<15} {'CMD'}")
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            cmd = f.read().replace(b'\0', b' ').decode().strip()
        if 'python' not in cmd or 'get_times.py' in cmd or 'multiprocessing' in cmd or 'bash' in cmd or 'proton' in cmd or 'glances' in cmd or 'networkd' in cmd or 'execsnoop' in cmd or 'printer' in cmd or 'pop-transition' in cmd: continue
        with open(f'/proc/{pid}/stat') as f:
            stat = f.read().split()
            starttime = int(stat[21]) / HZ
            running_time = time.time() - (btime + starttime)
            td = str(timedelta(seconds=int(running_time)))
            print(f"{pid:<8} {td:<15} {cmd}")
    except:
        pass
