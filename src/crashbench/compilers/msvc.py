from functools import cached_property
from .compiler import Compiler

class MSVC(Compiler):
    default_option_separator = ':'
    option_prefix = '/'
    @cached_property
    def discover(self):
        return []

    # @builtin
    # def error_code(self, code: int):
    #     ...