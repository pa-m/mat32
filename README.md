# mat32 module

adapted from Gonum matrix [![GoDoc](https://godoc.org/gonum.org/v1/gonum/mat?status.svg)](https://godoc.org/gonum.org/v1/gonum/mat)

- removed non-easy portable parts (lot of funcs dependent from lapack64, io stuff). 
- modified tolerances for tests since float32 operations have a lower precision.

pull requests will be happily considered.

