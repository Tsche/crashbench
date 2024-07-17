from .compiler import Compiler
from .clang import Clang
from .gcc import GCC
from .msvc import MSVC

compilers = [Clang, GCC, MSVC]
__all__ = ["Compiler", "Clang", "GCC", "MSVC"]

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
