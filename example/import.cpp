[[import("string", "ascii_lowercase", alias="ASCII")]];
[[import("string", {"digits", "ascii_uppercase"})]];

int main() {
    [[test("foo")]] {
        char const* ascii = ASCII;

        [[DIGITS::map(int_[3], digits)]];
        char map = DIGITS;
    }
}