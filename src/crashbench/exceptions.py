from typing import Optional
from tree_sitter import Node


class ParseError(Exception):
    def __init__(self, message: str, node: Optional[Node] = None):
        super().__init__(message)
        self.message = message
        self.node = node

class UndefinedError(Exception):
    def __init__(self, message: str, node: Node, name: str, scope, 
                 is_function: Optional[bool] = None):
        super().__init__(message)
        self.message = message
        self.node = node
        self.name = name
        self.scope = scope
        self.is_function = is_function
