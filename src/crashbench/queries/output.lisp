(attributed_statement 
    (attribute_declaration
        (attribute
            name: (identifier) @kind (.match? @kind "^(output)")
        ))
    (compound_statement) @parameters)