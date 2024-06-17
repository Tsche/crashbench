(attributed_statement
    (attribute_declaration 
        (attribute name: ((identifier) @using) (.eq? @using "using") )
    )
    .
    (labeled_statement
        label: (statement_identifier) @name
        (expression_statement
            [
                (comma_expression) @function
                (call_expression) @value
                (identifier) @value
            ]
        )
    )
)@node