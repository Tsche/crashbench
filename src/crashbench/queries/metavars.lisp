[(translation_unit (attributed_statement
            (attribute_declaration
                (attribute
                    prefix: (identifier) @name
                    name: (identifier) @generator
                    (argument_list)? @arguments 
        ))+
        (expression_statement ";")) @variable)
(function_definition
    (compound_statement
        (attributed_statement
            (attribute_declaration
                (attribute
                    prefix: (identifier) @name
                    name: (identifier) @generator
                    (argument_list)? @arguments 
        ))+
        (expression_statement ";")) @variable
))
(namespace_definition 
    body: (declaration_list
        (attributed_statement
            (attribute_declaration
                (attribute
                    prefix: (identifier) @name
                    name: (identifier) @generator
                    (argument_list)? @arguments 
        ))+
        (expression_statement ";")) @variable
))
]