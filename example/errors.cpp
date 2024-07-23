[[using summate: iterable, foldl(add, 0, iterable)]];
[[FOO::list(1, 2, 3, 4, 5)]];

int main(){
    [[test("foo")]] {
        [[BAR::summate(FOO)]];

        // mark BAR as used to force evaluation
        [[use(BAR)]];
    }
}
