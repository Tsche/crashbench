from .compiler import Compiler, builtin, COMPILER_BUILTINS
from .clang import Clang
from .gcc import GCC
#from .msvc import MSVC

COMPILERS = [Clang, GCC
             #, MSVC
            ]
__all__ = ["Compiler", "Clang", "GCC", "builtin", "COMPILER_BUILTINS"
           #"MSVC"
           ]

# def discover():
#     for compiler in COMPILERS:
#         yield from compiler().detected

def is_valid_compiler(name: str):
    return name in [compiler.__name__ for compiler in COMPILERS]

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
