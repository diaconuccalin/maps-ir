# maps-ir

`maps-ir` is an out-of-tree MLIR project for the MAPS tool. 

## Prerequisites

- CMake 3.20 or newer
- A C++17 compiler
- LLVM and MLIR development packages built with CMake package files

The build uses normal CMake package discovery for LLVM and MLIR. Point CMake at
your MLIR install with either `CMAKE_PREFIX_PATH`, or explicit `MLIR_DIR` and
`LLVM_DIR` values.

For an LLVM/MLIR install rooted at `/path/to/llvm-install`, the package
directories are typically:

- `/path/to/llvm-install/lib/cmake/llvm`
- `/path/to/llvm-install/lib/cmake/mlir`

## Build

Configure the MLIR project once:

```bash
cmake -S . -B build \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm
```

Build the local translator:

```bash
make maps-translate
```

The Makefile target wraps:

```bash
cmake --build build --target maps-translate
```

You can also build directly with CMake if preferred.

## Run The Example

Generate MAPS MLIR, the MAGIA header, and the MAGIA data file from the bundled
example JSON:

```bash
make magia-example
```

By default this reads:

```text
examples/magia_example.pipeline.json
```

and writes:

```text
generated/magia_example.pipeline.mlir
generated/magia_example.h
generated/magia_example_data.c
```

Run individual steps with:

```bash
make maps-mlir
make magia-header
make magia-data
```

The generated MAGIA names and token settings can be overridden:

```bash
make magia-example HEADER_NAME=debug_magia_example NUM_TOKENS=4 NUM_TOKEN_SLOTS=2
```

Remove generated translator outputs with:

```bash
make clean-generated
```


