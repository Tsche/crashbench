from functools import cached_property
from .compiler import Compiler

class MSVC(Compiler):
    @cached_property
    def discover(self):
        return []

    # @builtin
    # def error_code(self, code: int):
    #     ...