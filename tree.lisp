
(translation_unit 
    (function_definition type: 
        (primitive_type) declarator: 
        (function_declarator declarator: 
            (identifier) parameters: 
            (parameter_list)) body: 
        (compound_statement 
            (attributed_statement 
                (attribute_declaration 
                    (attribute name: 
                        (identifier) 
                        (argument_list 
                            (string_literal 
                                (string_content))))) 
                (compound_statement 
                    (attributed_statement 
                        (attribute_declaration 
                            (attribute prefix: 
                                (identifier) name: 
                                (identifier) 
                                (argument_list 
                                    (number_literal) 
                                    (number_literal)))) 
                        (expression_statement)) 
                    (expression_statement 
                        (call_expression function: 
                            (template_function name: 
                                (identifier) arguments: 
                                (template_argument_list 
                                    (type_descriptor type: 
                                        (type_identifier)))) arguments: 
                            (argument_list))))) 
            (attributed_statement 
                (attribute_declaration 
                    (attribute name: 
                        (identifier) 
                        (argument_list 
                            (string_literal 
                                (string_content))))) 
                (compound_statement 
                    (attributed_statement 
                        (attribute_declaration 
                            (attribute prefix: 
                                (identifier) name: 
                                (identifier) 
                                (argument_list 
                                    (string_literal 
                                        (string_content)) 
                                    (string_literal 
                                        (string_content))))) 
                        (expression_statement)) 
                    (expression_statement 
                        (compound_literal_expression type: 
                            (type_identifier) value: 
                            (initializer_list))))) 
            (attributed_statement 
                (attribute_declaration 
                    (attribute name: 
                        (identifier) 
                        (argument_list 
                            (string_literal 
                                (string_content))))) 
                (compound_statement 
                    (attributed_statement 
                        (attribute_declaration 
                            (attribute prefix: 
                                (identifier) name: 
                                (identifier) 
                                (argument_list 
                                    (string_literal 
                                        (string_content)) 
                                    (string_literal 
                                        (string_content))))) 
                        (expression_statement)) 
                    (expression_statement 
                        (call_expression function: 
                            (qualified_identifier scope: 
                                (namespace_identifier) name: 
                                (identifier)) arguments: 
                            (argument_list))))))) 
    (comment))
