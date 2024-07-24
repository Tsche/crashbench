from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import platform
from typing import Optional
from .compilers import CompilerFamily, discover
from .parser import TranslationUnit

compilers: list[CompilerFamily] = list(discover())

class Runner:
    def __init__(self, jobs: Optional[int] = None, pin_cpu: Optional[str] = None):
        self.pin_cpu = pin_cpu
        self.jobs = jobs
        self.build_dir = TemporaryDirectory("crashbench")

        if platform.system() != 'Linux' and pin_cpu is not None:
            logging.warning("CPU pinning is currently only supported for Linux. This setting will be ignored.")
            self.pin_cpu = None

    def run(self, source_path: Path):
        source = TranslationUnit(source_path)
        processed_path = Path(self.build_dir.name) / source_path.name
        processed_path.write_text(source.source)

        for test in source.tests:
            for variables in test.runs:
                for compiler in compilers:
                    print(compiler)
                    #compile_commands = compiler.compile_commands(processed_path, test.settings, variables)
                    #print(compile_commands)
