// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"github.com/pa-m/mat32/internal/asm/f32"
	"gonum.org/v1/gonum/blas/blas32"
)

// Inner computes the generalized inner product
//   x^T A y
// between column vectors x and y with matrix A. This is only a true inner product if
// A is symmetric positive definite, though the operation works for any matrix A.
//
// Inner panics if x.Len != m or y.Len != n when A is an m x n matrix.
func Inner(x Vector, a Matrix, y Vector) float32 {
	m, n := a.Dims()
	if x.Len() != m {
		panic(ErrShape)
	}
	if y.Len() != n {
		panic(ErrShape)
	}
	if m == 0 || n == 0 {
		return 0
	}

	var sum float32

	switch a := a.(type) {
	case RawMatrixer:
		amat := a.RawMatrix()
		var ymat blas32.Vector
		if yrv, ok := y.(RawVectorer); ok {
			ymat = yrv.RawVector()
		} else {
			break
		}
		for i := 0; i < x.Len(); i++ {
			xi := x.AtVec(i)
			if xi != 0 {
				if ymat.Inc == 1 {
					sum += xi * f32.DotUnitary(
						amat.Data[i*amat.Stride:i*amat.Stride+n],
						ymat.Data,
					)
				} else {
					sum += xi * f32.DotInc(
						amat.Data[i*amat.Stride:i*amat.Stride+n],
						ymat.Data, uintptr(n),
						1, uintptr(ymat.Inc),
						0, 0,
					)
				}
			}
		}
		return sum
	}
	for i := 0; i < x.Len(); i++ {
		xi := x.AtVec(i)
		for j := 0; j < y.Len(); j++ {
			sum += xi * a.At(i, j) * y.AtVec(j)
		}
	}
	return sum
}
