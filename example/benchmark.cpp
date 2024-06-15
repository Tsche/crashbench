template <int Idx [[range(0, 5)]], int [[range(10, 12)]] Bar>
[[benchmark]]
void test_something() {
  static_assert(Idx < 5, "foo");
}

[[benchmark]]
void test_namespace() {
  [[replace("foo", "bar")]] using namespace target;
}

// template <typename T>
// [[benchmark]]
// [[bind::T("int", "long")]]
// void test_type(){}

template <auto> void test() {}
namespace foo {
void zoinks() {}
} // namespace foo
namespace bar {
void zoinks() {}
} // namespace bar
namespace target {
void zoinks() {}
} // namespace target
struct Foo {};
struct Bar {};

int main() {
  [[benchmark("test_value")]] {
    static constexpr int Idx [[range(0, 10)]] = PARAM_IDX;
    test<Idx>();
  }
  [[benchmark("test_type")]] {
    using type [[replace("Foo", "Bar")]] = PARAM_TYPE;
    type{};
  }

  [[benchmark("test_namespace")]] {
    [[replace("foo", "bar")]] using namespace PARAM_TARGET;
    zoinks();
  }
}