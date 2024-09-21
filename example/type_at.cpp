#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace recursive {
template <std::size_t Idx, typename T, typename... Ts> struct GetImpl {
  using type = typename GetImpl<Idx - 1, Ts...>::type;
};

template <typename T, typename... Ts> struct GetImpl<0, T, Ts...> {
  using type = T;
};

template <std::size_t Idx, typename... Ts>
using get = typename GetImpl<Idx, Ts...>::type;
} // namespace recursive

namespace inheritance1 {
template <std::size_t Idx, typename T> struct Tagged {
  using type = T;
};

template <typename, typename...> struct GetHelper;

template <std::size_t... Idx, typename... Ts>
struct GetHelper<std::index_sequence<Idx...>, Ts...> : Tagged<Idx, Ts>... {};

template <std::size_t Idx, typename T> Tagged<Idx, T> get_impl(Tagged<Idx, T>);

template <std::size_t Idx, typename... Ts>
using get = typename decltype(get_impl<Idx>(
    GetHelper<std::index_sequence_for<Ts...>, Ts...>{}))::type;
} // namespace inheritance1

namespace inheritance2 {
template <std::size_t Idx, typename T> struct Tagged {
  using type = T;
  Tagged operator()(std::integral_constant<std::size_t, Idx>);
};

template <typename, typename...> struct GetHelper;

template <std::size_t... Idx, typename... Ts>
struct GetHelper<std::index_sequence<Idx...>, Ts...> : Tagged<Idx, Ts>... {
  using Tagged<Idx, Ts>::operator()...;
};

template <std::size_t Idx, typename... Ts>
using get = typename decltype(GetHelper<std::index_sequence_for<Ts...>, Ts...>{}(
    std::integral_constant<std::size_t, Idx>{}))::type;
} // namespace inheritance2

namespace voidptr {
template <std::size_t N, typename = std::make_index_sequence<N>> struct GetImpl;

template <std::size_t N, std::size_t... Skip>
struct GetImpl<N, std::index_sequence<Skip...>> {
  template <typename T>
  auto operator()(decltype((void *)Skip)..., T *, ...) -> T;
};

template <std::size_t Idx, typename... Ts>
using get = decltype(GetImpl<Idx>{}(static_cast<Ts *>(0)...));
} // namespace voidptr

namespace ignored {
struct Universal {
  constexpr Universal() = default;
  constexpr Universal(auto&&) {}
  constexpr Universal& operator=(auto&&) { return *this; }
};

template <std::size_t N, typename = std::make_index_sequence<N>> struct GetImpl;

template <std::size_t N, std::size_t... Skip>
struct GetImpl<N, std::index_sequence<Skip...>> {
  template <typename T>
  auto operator()(decltype(Universal{Skip})..., T&&, auto&&...) -> T;
};

template <std::size_t Idx, typename... Ts>
using get = decltype(GetImpl<Idx>{}(std::declval<Ts>()...));
} // namespace ignored

namespace paging {
template <std::size_t Idx, typename T> struct GetImpl;

template <template <typename...> class List, typename T0, typename... Ts>
struct GetImpl<0, List<T0, Ts...>> {
  using type = T0;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename... Ts>
struct GetImpl<1, List<T0, T1, Ts...>> {
  using type = T1;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename... Ts>
struct GetImpl<2, List<T0, T1, T2, Ts...>> {
  using type = T2;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename... Ts>
struct GetImpl<3, List<T0, T1, T2, T3, Ts...>> {
  using type = T3;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename... Ts>
struct GetImpl<4, List<T0, T1, T2, T3, T4, Ts...>> {
  using type = T4;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename... Ts>
struct GetImpl<5, List<T0, T1, T2, T3, T4, T5, Ts...>> {
  using type = T5;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename... Ts>
struct GetImpl<6, List<T0, T1, T2, T3, T4, T5, T6, Ts...>> {
  using type = T6;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename... Ts>
struct GetImpl<7, List<T0, T1, T2, T3, T4, T5, T6, T7, Ts...>> {
  using type = T7;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename... Ts>
struct GetImpl<8, List<T0, T1, T2, T3, T4, T5, T6, T7, T8, Ts...>> {
  using type = T8;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename... Ts>
struct GetImpl<9, List<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, Ts...>> {
  using type = T9;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename T10, typename... Ts>
struct GetImpl<10, List<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, Ts...>> {
  using type = T10;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename T10, typename T11,
          typename... Ts>
struct GetImpl<11,
               List<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, Ts...>> {
  using type = T11;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename T10, typename T11,
          typename T12, typename... Ts>
struct GetImpl<
    12, List<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, Ts...>> {
  using type = T12;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename T10, typename T11,
          typename T12, typename T13, typename... Ts>
struct GetImpl<13, List<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
                        T13, Ts...>> {
  using type = T13;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename T10, typename T11,
          typename T12, typename T13, typename T14, typename... Ts>
struct GetImpl<14, List<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
                        T13, T14, Ts...>> {
  using type = T14;
  using tail = List<Ts...>;
};

template <template <typename...> class List, typename T0, typename T1,
          typename T2, typename T3, typename T4, typename T5, typename T6,
          typename T7, typename T8, typename T9, typename T10, typename T11,
          typename T12, typename T13, typename T14, typename T15,
          typename... Ts>
struct GetImpl<15, List<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12,
                        T13, T14, T15, Ts...>> {
  using type = T15;
  using tail = List<Ts...>;
};

constexpr inline auto PAGE_SIZE = 15U;

template <std::size_t Idx, template <typename...> class List, typename... Ts>
  requires(Idx >= PAGE_SIZE)
struct GetImpl<Idx, List<Ts...>> {
  using first_page = GetImpl<PAGE_SIZE - 1, List<Ts...>>;
  using recurse = GetImpl<Idx - PAGE_SIZE, typename first_page::tail>;
  using type = typename recurse::type;
  using tail = typename recurse::tail;
};

template <typename...> struct List {};
template <std::size_t Idx, typename... Ts>
using get = typename GetImpl<Idx, List<Ts...>>::type;
} // namespace paging

namespace nested {
namespace detail {
template <std::size_t Idx, typename T> struct Tagged {
  using type = T;
  constexpr static std::size_t value = Idx;
};

template <std::size_t Idx, typename T>
constexpr T get_type_impl(Tagged<Idx, T>) {
  static_assert(false, "get_type_impl not allowed in an evaluated context");
}
} // namespace detail

template <typename... Ts> struct TypeList2 {
  template <typename = std::index_sequence_for<Ts...>> struct GetHelper;

  template <std::size_t... Idx>
  struct GetHelper<std::index_sequence<Idx...>> : detail::Tagged<Idx, Ts>... {};

  template <std::size_t Idx>
  using type_at = decltype(get_type_impl<Idx>(GetHelper{}));
};

template <std::size_t Idx, typename... Ts>
using get = typename TypeList2<Ts...>::template type_at<Idx>;
} // namespace nested

#if __has_builtin(__type_pack_element)
namespace builtin {
template <std::size_t Idx, typename... Ts>
using get = __type_pack_element<Idx, Ts...>;
} // namespace builtin
#else
#error "No pack index builtin detected"
#endif

// #ifdef __clang__
// // currently only clang supports this
// namespace cpp26 {
// template <std::size_t Idx, typename... Ts> using get = Ts...[Idx];
// } // namespace cpp26
// #endif

template <auto> struct Dummy {};

template <template <std::size_t, typename...> class getter, std::size_t... Idx>
void run(std::index_sequence<Idx...>) {
  static_assert((std::same_as<getter<Idx, Dummy<Idx>...>, Dummy<Idx>> && ...), "");
}

[[language("c++")]];
[[standard("c++23")]];

int main() {
  [[benchmark("type_at")]] {
    [[using STRATEGY: list("recursive", "inheritance1", "inheritance2", "voidptr", "ignored", "nested", "paging", "builtin" /*, "cpp26"*/)]];
    [[using COUNT:    range(1, 255)]];

    run<STRATEGY::get>(std::make_index_sequence<COUNT>{});

    [[plot]]{
      [[using draw_line: LABEL, step(x=COUNT, y=$.elapsed_ms, legend_label=LABEL, mode="center")]];
      [[using draw_plot: TITLE, LINES, figure(title=TITLE,
                                              width=1200, 
                                              height=800, 
                                              x_axis_label="Count", 
                                              y_axis_label="Translation Time (ms)",
                                              data=LINES)]];
      
      [[using LINES: map(draw_line, STRATEGY)]];
      [[using create_plot: TITLE, draw_plot(TITLE, LINES)]];
      [[using create_tab: COMPILER, tab(child=create_plot(COMPILER), title=COMPILER)]];
      [[using TABS: map(create_tab, $.compilers)]];
      [[draw(TABS)]];
    }
  }
}
