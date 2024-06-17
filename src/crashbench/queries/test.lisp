(translation_unit 
    [
        (attributed_statement
            (attribute_declaration (attribute) @attr)+
            . (expression_statement ";")) @attr_node

        (attributed_statement
            (attribute_declaration (attribute name: ((identifier) @using) (.eq? @using "using")))
            . (labeled_statement)
        ) @attr_node

        (namespace_definition 
            body: (declaration_list
                [
                    (attributed_statement
                        (attribute_declaration (attribute) @attr)+
                        . (expression_statement ";")) @attr_node

                    (attributed_statement
                        (attribute_declaration (attribute name: ((identifier) @using) (.eq? @using "using")))
                        . (labeled_statement)
                    ) @attr_node
                ]
        ))

        (function_definition
            (compound_statement
                [
                    (attributed_statement
                        (attribute_declaration (attribute) @attr)+
                        (expression_statement ";")) @attr_node

                    (attributed_statement
                        (attribute_declaration (attribute name: ((identifier) @using) (.eq? @using "using")))
                        . (labeled_statement)
                    ) @attr_node

                    (attributed_statement 
                        (attribute_declaration
                            (attribute
                                name: (identifier) @kind (.match? @kind "^(benchmark|test)")
                                (argument_list (string_literal (string_content) @test_name))
                        ) ) @test_head
                        . (compound_statement
                            [
                                (attributed_statement
                                    (attribute_declaration)+
                                    . (expression_statement ";")) @attr_node

                                (attributed_statement
                                    (attribute_declaration (attribute name: ((identifier) @using) (.eq? @using "using")))
                                    . (labeled_statement)
                                ) @attr_node
                                (_)
                            ]*
                        ) @code)
                ]
            )
        )
    ]
)