[[using fib: n, return(n <= 1 ? n : fib(n - 1) + fib(n - 2))]];
[[test_range::range(10)]];
[[fibonacci::fib]];
int main(){
    [[test("foo")]] {
        [[BAR::map(fibonacci, test_range)]];
        [[FOO::list(true, false)]];
        // [[use(FOO)]];
        int x = BAR;
    }
}
