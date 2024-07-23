[[using fib: n, if(le(n, 1), n, add(fib(sub(n, 1)), fib(sub(n, 2))))]];
[[test_range::range(10)]];

int main(){
    [[test("foo")]] {
        [[BAR::map(fib, test_range)]];
        [[use(BAR)]];
    }
}
