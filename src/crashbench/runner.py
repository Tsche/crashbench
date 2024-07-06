from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import platform
from typing import Any, Optional
from .compilers import Compiler, discover
from .parser import TranslationUnit

def transform_metavar(name: str, value: Any):
    if isinstance(value, str) and " " in value:
        # wrap strings with spaces in them
        value = f'"{value}"'
    return f"-D{name}={value}"

compilers: list[Compiler] = list(discover())

class Runner:
    def __init__(self, pin_cpu: Optional[int] = None):
        self.pin_cpu = pin_cpu
        self.build_dir = TemporaryDirectory("crashbench") # TODO use context manager

        if platform.system() != 'Linux' and pin_cpu is not None:
            logging.warning("CPU pinning is currently only supported for Linux. This setting will be ignored.")
            self.pin_cpu = None

    def run(self, source_path: Path):
        source = TranslationUnit(source_path)
        processed_path = self.build_dir / source_path.name
        processed_path.write_bytes(source.source)

        for test in source.tests:
            for variables in test.runs:
                for compiler in compilers:
                    yield compiler.compile_commands(processed_path, test.settings, variables)
