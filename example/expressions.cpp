int main(){
    [[test("field_expr")]]{
        [[using foo: x, return(x.upper())]];
        [[using bar: foo("bar")]];
        [[use(bar)]];
    }

    [[test("subscript")]]{
        [[using foo: x, return(x[2])]];
        [[using bar: foo("bar")]];
        [[use(bar)]];
    }
}