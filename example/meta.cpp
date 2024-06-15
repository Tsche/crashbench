[[META1::range(5)]];

template<typename T>
void foo(){
    [[META2::range(5)]];
}

namespace Bar {
    [[META4::range(5)]];
}

int main(){
    [[META5::range(5)]];
    [[benchmark("test")]] {
        [[Var::map(bind(concat, "typename "), map(str, range(0, 5)))]];
    }
}