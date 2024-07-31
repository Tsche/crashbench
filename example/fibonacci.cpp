[[using fib: n, return(n <= 1 ? n : fib(n - 1) + fib(n - 2))]];
[[test_range::range(10)]];
[[BAR::var(1)]];
int main(){
    [[test("foo")]] {
        [[BAR::map(fib, test_range)]];
        int x = BAR;
    }
}
