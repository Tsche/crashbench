// [[language("c++")]];

// disables compilers with no gnu language extensions support (msvc)
// sets gnu prefix for `-std=`
// TODO set std explicitly if `standard` is missing
// [[gnu_extensions(true)]];

// host is special -> host system architecture
// [[target("host")]];

// select first supported standard greater than 20
[[standard(">20")]]; 

// select all standards greater than 20
// [[standards(">20")]];

// compiler specific settings
// msvc implicitly disabled (msvc doesn't support gnu extensions)
// [[GCC(enabled=false)]]; // GCC disabled for this TU

// // versions for multiple versions?
// [[Clang(version=">=16")]];

int main(){
    [[benchmark("bench")]] {
        // [[standard("17")]];
        [[Clang::standard("17")]];
        // [[Clang::trace(false)]];
        [[BAR::var(true)]];
        [[FOO::range(5)]];
        static_assert(FOO <= 5);
        [[use(BAR)]];
    }
    [[test("foo")]] {
        // [[GCC(enabled=true)]]; // re-enable GCC for this test
        // [[Clang(version=">=12.0", trace=true)]];
        [[error("error text", regex=false)]];
        [[GCC::error("foo2")]];
        [[GCC::link(true)]];
    }
}