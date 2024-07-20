from typing import Optional
from .compilers import compilers

def is_valid_compiler(name: str):
    return name in [compiler.__name__ for compiler in compilers]

builtins = {}

@staticmethod
def builtin(fnc):
    builtins[fnc.__name__] = fnc
    return fnc

class Settings:
    def __init__(self, parent: Optional['Settings'] = None):
        # TODO default to parent settings

        self.compiler_settings = {}
        self.is_gnu = False
        self.architecture = "host"
        self.lang = "c++"
        self.dialect = "20"

    def __str__(self):
        lines = []
        lines.append(f"target:   {self.architecture}")
        lines.append(f"language: {self.lang}")
        lines.append(f"dialect:  {self.dialect}")
        lines.append(f"is_gnu:   {self.is_gnu}")
        for compiler, settings in self.compiler_settings.items():
            lines.append(f"{compiler}:")
            for setting in settings:
                lines.append(f"  {setting}")
        return '\n'.join(lines)

    def add(self, name, args, kwargs):
        assert name != "add"
        
        if name in builtins:
            builtins[name](self, *args, **kwargs)
            return

        if is_valid_compiler(name):
            self.compiler_setting(name, *args, **kwargs)
            return

        raise ValueError(f"Unrecognized setting {name}")

    @builtin
    def compiler_setting(self, compiler: str, *args, enabled: bool = True, **kwargs):
        assert is_valid_compiler(compiler), f"Unrecognized compiler {compiler}"
        # print(enabled)
        # print(args)
        # print(kwargs)

    @builtin
    def gnu_extensions(self, enabled: bool):
        self.is_gnu = True

    @builtin
    def target(self, value: str):
        if value == "host":
            # use host architecture
            ...
        self.architecture = value

    @builtin
    def language(self, *lang: str):
        if len(lang) == 1:
            self.lang = lang[0]
        elif len(lang) > 1:
            self.lang = list(lang)
        else:
            raise ValueError("language needs at least one argument")

    @builtin
    def standard(self, selector: str):
        ...

    @builtin
    def standards(self, minimum: str, maximum: Optional[str] = None):
        ...