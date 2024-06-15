
[[deprecated]]
[[test]]
void test_something(){
    [[assert_failure("foo")]];
    static_assert(false, "foo");
}
