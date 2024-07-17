from .compiler import Compiler

class MSVC(Compiler):
    @classmethod
    def discover(cls):
        return
        yield