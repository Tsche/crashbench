(attributed_statement
    (attribute_declaration
        (attribute
            prefix: (identifier) @name
            name: (identifier) @generator
            (argument_list)? @arguments 
        ))+
    (expression_statement ";")) @variable