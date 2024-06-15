
(attributed_statement 
    (attribute_declaration
        (attribute
            name: (identifier) @kind (.match? @kind "^(benchmark|test)")
            (argument_list
                (string_literal 
                    (string_content) @name))
        ) ) @attributes
    (compound_statement) @code)