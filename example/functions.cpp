[[using SQUARE: x, mul(x, x)]];
[[using PREFIX: pre, lst, map(bind(concat, pre), map(str, lst))]];

int main() {
    [[FOO::list(3, 2)]];
    [[TMP::map(SQUARE, FOO)]];
    [[RET::foldl(add, 0, TMP)]];

    [[TEST::PREFIX("foo_", FOO)]];

    [[benchmark("test_name")]] {
        [[use(TEST)]];
        int x = RET;
    }
}
