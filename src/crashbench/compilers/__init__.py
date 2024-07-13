import platform

from .compiler import Compiler

compilers = []
__all__ = ["Compiler"]

match platform.system():
    case "Linux":
        from .clang import Clang
        from .gcc import GCC

        compilers = [Clang, GCC]
        __all__.extend(["Clang", "GCC"])

    case "Windows":
        from .clang import Clang
        from .gcc import GCC
        from .msvc import MSVC

        compilers = [Clang, GCC, MSVC]
        __all__.extend(["Clang", "GCC", "MSVC"])

    case "Darwin":
        from .clang import Clang

        compilers = [Clang]
        __all__.extend(["Clang"])


def discover():
    for compiler in compilers:
        yield from compiler.discover()


"""
SARIF:

clang >=16
-fdiagnostics-format=sarif
always stderr

gcc >=13
-fdiagnostics-format=sarif-stderr
-fdiagnostics-format=sarif-file

msvc
/experimental:log{file_stem}
"""
