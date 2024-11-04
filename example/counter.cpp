#include <cstdint>
#include <utility>

// implementations

namespace simple {
template <typename Key> 
struct Counter {
  static constexpr std::size_t npos = 0ULL - 1;
  template <std::size_t N> struct Value {
    friend auto adl_shenanigans(Value);
    struct Set {
      friend auto adl_shenanigans(Value) {}
    };
  };

  template <auto Tag = [] {}, std::size_t N = 0>
  static consteval std::size_t current() {
    if constexpr (requires(Value<N> value) { adl_shenanigans(value); }) {
      return current<Tag, N + 1>();
    } else {
      return N - 1;
    }
  }

  template <auto Tag = [] {}> static consteval std::size_t next() {
    constexpr std::size_t last = current<Tag>();
    (void)typename Value<last + 1>::Set{};
    return last + 1;
  }
};
} // namespace simple

namespace determine_range_first {
template <typename Key, std::size_t Start = 0, std::size_t Step = 1>
struct Counter {
  static constexpr std::size_t npos = 0ULL - Step;

  template <std::size_t N> struct Value {
    friend auto adl_shenanigans(Value);
    struct Set {
      friend auto adl_shenanigans(Value) {}
    };
  };

  template <std::size_t Min, std::size_t Max, std::size_t Increment = Step,
            typename = std::make_index_sequence<((Max - Min) / Increment) + 1>>
  struct Range;

  template <std::size_t Min, std::size_t Max, std::size_t Increment,
            std::size_t... Idx>
  struct Range<Min, Max, Increment, std::index_sequence<Idx...>> {
    static constexpr std::size_t values[] = {Idx...};

    template <auto Tag = [] {}> static constexpr std::size_t largest() {
      int index = -1;
      if ((... && (++index, requires (Value<Min + Increment * Idx> value) { adl_shenanigans(value); }))){
        // not in this range, return one past the end
        return Min + Increment * index;
      }

      if (index <= 0) {
        return npos;
      }

      return Min + Increment * (index - 1);
    }
  };

  template <auto Tag = [] {}, std::size_t N = Start>
  static consteval std::size_t current() {
    constexpr std::size_t probe_amount = 100;
    constexpr std::size_t range_min =
        Range<N, N + probe_amount * probe_amount,
              probe_amount>::template largest<Tag>();
    if (range_min == npos) {
      return npos;
    } else if (range_min == N + probe_amount * probe_amount) {
      return current<Tag, range_min>();
    }

    return Range<range_min, range_min + probe_amount>::largest();
  }

  template <auto Tag = [] {}> static consteval std::size_t next() {
    constexpr std::size_t next_value = current<Tag>() + Step;
    (void)typename Value<next_value>::Set{};
    return next_value;
  }
};
} // namespace determine_range_first

namespace search_ranges {
template <typename Key>
struct Counter {
  static constexpr std::size_t npos = 0ULL - 1;

  template <std::size_t N> struct Value {
    friend auto adl_shenanigans(Value);
    struct Set {
      friend auto adl_shenanigans(Value) {}
    };
  };

  template <std::size_t Min, std::size_t Max, typename = std::make_index_sequence<(Max - Min) + 1>>
  struct Range;

  template <std::size_t Min, std::size_t Max, std::size_t... Idx>
  struct Range<Min, Max, std::index_sequence<Idx...>> {
    template <auto Tag = [] {}> static constexpr std::size_t largest() {
      int index = -1;
      if ((... && (++index, requires (Value<Min + Idx> value) { adl_shenanigans(value); }))){
        // not in this range, return one past the end
        return Min + index;
      }

      if (index <= 0) {
        return npos;
      }

      return Min + (index - 1);
    }
  };

  template <auto Tag = [] {}, std::size_t N = 0>
  static consteval std::size_t current() {
    constexpr std::size_t probe_amount = 50;
    constexpr std::size_t range_min = Range<N, N + probe_amount>::template largest<Tag>();
    if constexpr (range_min == npos) {
      return npos;
    } else if constexpr (range_min == N + probe_amount) {
      return current<Tag, range_min>();
    }
    else {
        return range_min;
    }
  }

  template <auto Tag = [] {}> static consteval std::size_t next() {
    constexpr std::size_t next_value = current<Tag>() + 1;
    (void)typename Value<next_value>::Set{};
    return next_value;
  }
};
}


namespace array {
template <typename Key>
struct Counter {
  static constexpr std::size_t npos = 0ULL - 1;

  template <std::size_t N> struct Value {
    friend auto adl_shenanigans(Value);
    struct Set {
      friend auto adl_shenanigans(Value) {}
    };
  };

  template <std::size_t Min, std::size_t Max, typename = std::make_index_sequence<(Max - Min) + 1>>
  struct Range;

  template <std::size_t Min, std::size_t Max, std::size_t... Idx>
  struct Range<Min, Max, std::index_sequence<Idx...>> {
    template <auto Tag = [] {}> static constexpr std::size_t largest() {
      std::size_t values[] = {requires (Value<Min + Idx> value) { adl_shenanigans(value); }...};
      for (std::size_t idx = 0; idx <= Max - Min; ++idx){
        if (!values[idx]){
            return Min + idx - 1;
        }
      }
      return Max;
    }
  };

  template <auto Tag = [] {}, std::size_t N = 0>
  static consteval std::size_t current() {
    constexpr std::size_t probe_amount = 100;
    constexpr std::size_t range_min = Range<N, N + probe_amount>::template largest<Tag>();
    if constexpr (range_min == npos) {
      return npos;
    } else if constexpr (range_min == N + probe_amount) {
      return current<Tag, range_min>();
    }
    else {
        return range_min;
    }
  }

  template <auto Tag = [] {}> static consteval std::size_t next() {
    constexpr std::size_t next_value = current<Tag>() + 1;
    (void)typename Value<next_value>::Set{};
    return next_value;
  }
};
}

// Tests
#pragma clang diagnostic ignored "-Wunused-value"
#pragma GCC diagnostic ignored "-Wnon-template-friend"
template <std::size_t Count, typename C, auto Offset = 0>
constexpr std::size_t increase_by() {
  []<std::size_t... Idx>(std::index_sequence<Idx...>) {
    ((C::template next<Offset + Idx>(), Idx), ...);
  }(std::make_index_sequence<Count>{});
  return C::current();
}

template <typename C> void test() {
  static_assert(C::current() == C::npos);
  static_assert(C::next() == 0);
  static_assert(C::current() == 0);
  static_assert(increase_by<50, C>() == 50);
  static_assert(increase_by<250, C, C::current()>() == 300);
  static_assert(C::next() == 301);
  static_assert(C::current() == 301);
}

[[language("c++")]];
[[standard("c++23")]];
// [[Clang::trace(true)]];

int main() {
  [[benchmark("counter")]] {
    [[using STRATEGY: list("determine_range_first", "simple", "search_ranges", "array")]];
    [[using COUNT: var(0)]];
    [[use(COUNT)]];
    test<STRATEGY::Counter<int>>();

    using counter2 = STRATEGY::Counter<float>;
    static_assert(counter2::next() == 0);
  }
}
