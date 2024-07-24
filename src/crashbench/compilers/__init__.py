from .compiler import CompilerFamily
from .clang import Clang
from .gcc import GCC
#from .msvc import MSVC

compilers = [Clang, GCC
             #, MSVC
            ]
__all__ = ["CompilerFamily", "Clang", "GCC", 
           #"MSVC"
           ]

def discover():
    for compiler in compilers:
        yield from compiler().detected

def is_valid_compiler(name: str):
    return name in [compiler.__name__ for compiler in compilers]

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
