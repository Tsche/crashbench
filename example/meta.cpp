
// // [[META1::range(5)]] [[META::var(2)]];
// [[compiler("gcc>=14", "clang>=18", "msvc")]];

// template<typename T>
// void foo(){
//     [[META2::range(5)]];
// }

namespace Bar {
    // [[META4::range(5)]];
}

int main(){
    // [[META5::range(5)]];
    // [[benchmark("test")]] {
    //     [[standard(">=26")]];
    //     // [[Var::map(bind(concat, "typename "), map(str, range(0, 5)))]];

    //     // [[use(Var)]];
    // }
    // [[Test::range(4)]] [[Test2::range(2)]];
    // [[TE::range(3)]] [[R::var(join(',', map(str, TE)))]];
    // [[TE::range(3)]] [[R::map(str, TE)]];
    // [[META6::var(2)]];

    [[FOO::var(5, 2, 3)]];
    [[test("foo")]] {
        [[R::map(bind(add, 2), FOO)]];

        [[use(R)]];
    }
}