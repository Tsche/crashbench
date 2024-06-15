template <
    typename TP,
    // (type_parameter_declaration (type_identifier))
    class TP2,
    // (type_parameter_declaration (type_identifier))
    int NTTP,
    // (parameter_declaration type: (primitive_type) declarator: (identifier)) 
    auto NTTP2,
    // (parameter_declaration type: (placeholder_type_specifier (auto)) declarator: (identifier)) 
    template <typename> class TTP,
    /*
        (template_template_parameter_declaration parameters: 
            (template_parameter_list 
                (type_parameter_declaration)) 
            (type_parameter_declaration 
                (type_identifier))) 
    */
    template <typename...> class TTP2,
    /*
        (template_template_parameter_declaration parameters: 
            (template_parameter_list 
                (variadic_type_parameter_declaration)) 
            (type_parameter_declaration 
                (type_identifier))) 
    */
    template <auto...> class TTP3,
    /*
        (template_template_parameter_declaration parameters: 
            (template_parameter_list 
                (variadic_parameter_declaration type: 
                    (placeholder_type_specifier 
                        (auto)) declarator: 
                    (variadic_declarator))) 
            (type_parameter_declaration 
                (type_identifier))) 
    */
    template <auto...> class TTP4 = Bar,
    /*
    (template_template_parameter_declaration parameters: 
                (template_parameter_list 
                    (variadic_parameter_declaration type: 
                        (placeholder_type_specifier 
                            (auto)) declarator: 
                        (variadic_declarator))) 
                (optional_type_parameter_declaration name: 
                    (type_identifier) default_type: 
                    (type_identifier)))
    */
    typename TP3 = int,
    // (optional_type_parameter_declaration name: (type_identifier) default_type: (primitive_type)) 
    int NTTP3 = 3,
    // (optional_parameter_declaration type: (primitive_type) declarator: (identifier) default_value: (number_literal)) 
    typename... TPP,
    // (variadic_type_parameter_declaration (type_identifier)) 
    int... NTTPP,
    // (variadic_parameter_declaration type: (primitive_type) declarator: (variadic_declarator (identifier))) 
    auto... NTTPP2
    /*
        (variadic_parameter_declaration type: 
            (placeholder_type_specifier 
                (auto)) declarator: 
            (variadic_declarator 
                (identifier)))) 
    */>
struct Foo;
