from functools import cached_property
from .compiler import CompilerFamily

class MSVC(CompilerFamily):
    @cached_property
    def detected(self):
        return []

    # @builtin
    # def error_code(self, code: int):
    #     ...