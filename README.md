# Experiments with metrics in embedding spaces

For use in the Semantic Kernel and elsewhere.

To check the hardware supports the necessary features, run `list_avx_features.sh` and look for "avx512_bf16" and "avx512_fp16"

## Prepare an Ubuntu machine to run these kernels

1. Add the following lines to `/etc/apt/sources.list.d/llvm-latest.list

```
deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main
deb-src http://apt.llvm.org/jammy/ llvm-toolchain-jammy-16 main
```

1. Install prerequisites to set up llvm repos
```sh
apt install wget gnupg
```

1. Add llvm repo key to `apt` (note that `apt-key` is deprecated, I should update this)
```sh
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
```

1. Update list of available packages and install `clang`

```sh
apt update
apt install clang-16 lldb-16 lld-16
```

1. `cmake` and `conan` are used here, but oneCloud machines already have `cmake` and `conda` installed.  If it's not

```sh
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
chmod a+x Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
./Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b
export PATH=$HOME/miniconda3/bin:$PATH
conda update conda
```

1. Use a `conan` profile similar to this:

```
[settings]
arch=x86_64
build_type=Release
os=Linux
compiler=clang
compiler.libcxx=libstdc++
compiler.version=16

[conf]
tools.build:compiler_executables={"c":"clang-16", "cpp":"clang++-16"}
```