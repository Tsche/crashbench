from dataclasses import dataclass
import os
import platform
import re
import subprocess
from typing import TypeAlias

import psutil

from .util import fnv1a

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        matches = re.search(r"model name\s*:\s(?P<label>.*)", all_info)
        assert matches is not None, "No CPU model name matches"
        return matches.groups()[0]
    return ""

@dataclass
class SystemInfo:
    architecture: str
    os_name: str
    os_version: str
    cpu_name: str
    cpu_count: int
    ram_amount: int

    def __init__(self):
        uname = platform.uname()
        self.architecture = uname.machine

        self.cpu_name = get_processor_name()
        self.os_name = uname.system
        self.os_version = uname.release
        self.cpu_count = psutil.cpu_count(logical=False)
        self.ram_amount = psutil.virtual_memory().total

SystemInfoHash: TypeAlias = int

SYSTEM_INFO = SystemInfo()
SYSTEM_INFO_HASH = fnv1a(SYSTEM_INFO)
