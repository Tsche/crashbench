// fib :: Int -> Int
// fib n = if n <= 1 then n else fib (n - 1) + fib (n - 2)
[[using fib: n, return(n <= 1 ? n : fib(n - 1) + fib(n - 2))]];

// test_range = [0..9]
[[using test_range: range(10)]];

// BAR = 1
[[using BAR: var(1)]];

int main() {
  // opens new scope
  [[test("foo")]] {

    // BAR = map fib test_range -- this is okay, BAR from the outer scope will be shadowed
    [[using BAR: map(fib, test_range)]];

    // sets BAR as used => forces evaluation
    int x = BAR;
  }
}