from typing import Optional
from .compilers import compilers


# TODO compiler specific settings

def is_valid_compiler(name: str):
    return name in [compiler.name for compiler in compilers]


class Settings:
    @staticmethod
    def compiler_setting(compiler: str, *args, **kwargs):
        assert is_valid_compiler(compiler), f"Unrecognized compiler {compiler}"

    @staticmethod
    def gnu_extensions(enabled: bool):
        ...

    @staticmethod
    def target(value: str):
        if value == "host":
            # use host architecture
            ...

    @staticmethod
    def language(*lang: str):
        ...

    @staticmethod
    def standard(selector: str):
        ...

    @staticmethod
    def standards(minimum: str, maximum: Optional[str] = None):
        ...