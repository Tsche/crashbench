int main() {
//   [[benchmark("test_value")]] {
//     [[Idx::range(0, 5)]];
//     test<Idx>();
//   }
//   [[benchmark("test_type")]] {
//     [[T::var("Foo", "Bar")]];
//     T{};
//   }

  [[benchmark("test_namespace")]] {
    // [[BS::var('c', "bar""baz", 1, true, false, nullptr, -2, 2.3, var(1, 2))]];
    [[NS::var("foo"/*, "bar"*/)]];
    // [[T::range(0, 4)]];
    // [[Foo::range(0, 2)]];

    NS::zoinks();
    BS::boings();
  }
}
