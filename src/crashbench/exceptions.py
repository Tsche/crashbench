from pathlib import Path
from tree_sitter import Node


class ParseError(Exception):
    def __init__(self, message: str, node: Node):
        super().__init__(message)
        self.message = message
        self.node = node

class UndefinedError(Exception):
    def __init__(self, message: str, node: Node, name: str, scope):
        super().__init__(message)
        self.message = message
        self.node = node
        self.name = name
        self.scope = scope
