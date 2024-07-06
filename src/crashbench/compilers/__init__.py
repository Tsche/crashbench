import platform

from .compiler import Compiler
from .clang import Clang
from .gcc import GCC

compilers = [Clang, GCC]
__all__ = ['Compiler', 'Clang', 'GCC']
if platform.system() == 'Windows':
    from .msvc import MSVC
    compilers.append(MSVC)
    __all__.append('MSVC')

def discover():
    for compiler in compilers:
        yield from compiler.discover()

'''
SARIF:

clang >=16
-fdiagnostics-format=sarif
always stderr

gcc >=13
-fdiagnostics-format=sarif-stderr
-fdiagnostics-format=sarif-file

msvc
/experimental:log{file_stem}
'''