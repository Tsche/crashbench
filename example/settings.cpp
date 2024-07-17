[[language("c++")]];

// disables compilers with no gnu language extensions support (msvc)
// sets gnu prefix for `-std=`
// TODO set std explicitly if `standard` is missing
[[gnu_extensions(true)]];

// host is special -> host system architecture
[[target("host")]];

// select first supported standard greater than 20
[[standard(">20")]]; 

// select all standards greater than 20
// [[standards(">20")]];

// compiler specific settings
// msvc implicitly disabled (msvc doesn't support gnu extensions)
[[GCC(false)]]; // GCC disabled for this TU

// versions for multiple versions?
// [[Clang(version=">=18", trace=true)]];

[[x::var(value=3)]];