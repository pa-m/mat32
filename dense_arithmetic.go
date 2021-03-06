// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

// Add adds a and b element-wise, placing the result in the receiver. Add
// will panic if the two matrices do not have the same shape.
func (m *Dense) Add(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}

	aU, _ := untranspose(a)
	bU, _ := untranspose(b)
	m.reuseAs(ar, ac)

	if arm, ok := a.(RawMatrixer); ok {
		if brm, ok := b.(RawMatrixer); ok {
			amat, bmat := arm.RawMatrix(), brm.RawMatrix()
			if m != aU {
				m.checkOverlap(amat)
			}
			if m != bU {
				m.checkOverlap(bmat)
			}
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v + bmat.Data[i+jb]
				}
			}
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)+b.At(r, c))
		}
	}
}

// Sub subtracts the matrix b from a, placing the result in the receiver. Sub
// will panic if the two matrices do not have the same shape.
func (m *Dense) Sub(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}

	aU, _ := untranspose(a)
	bU, _ := untranspose(b)
	m.reuseAs(ar, ac)

	if arm, ok := a.(RawMatrixer); ok {
		if brm, ok := b.(RawMatrixer); ok {
			amat, bmat := arm.RawMatrix(), brm.RawMatrix()
			if m != aU {
				m.checkOverlap(amat)
			}
			if m != bU {
				m.checkOverlap(bmat)
			}
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v - bmat.Data[i+jb]
				}
			}
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)-b.At(r, c))
		}
	}
}

// MulElem performs element-wise multiplication of a and b, placing the result
// in the receiver. MulElem will panic if the two matrices do not have the same
// shape.
func (m *Dense) MulElem(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}

	aU, _ := untranspose(a)
	bU, _ := untranspose(b)
	m.reuseAs(ar, ac)

	if arm, ok := a.(RawMatrixer); ok {
		if brm, ok := b.(RawMatrixer); ok {
			amat, bmat := arm.RawMatrix(), brm.RawMatrix()
			if m != aU {
				m.checkOverlap(amat)
			}
			if m != bU {
				m.checkOverlap(bmat)
			}
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v * bmat.Data[i+jb]
				}
			}
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)*b.At(r, c))
		}
	}
}

// DivElem performs element-wise division of a by b, placing the result
// in the receiver. DivElem will panic if the two matrices do not have the same
// shape.
func (m *Dense) DivElem(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		panic(ErrShape)
	}

	aU, _ := untranspose(a)
	bU, _ := untranspose(b)
	m.reuseAs(ar, ac)

	if arm, ok := a.(RawMatrixer); ok {
		if brm, ok := b.(RawMatrixer); ok {
			amat, bmat := arm.RawMatrix(), brm.RawMatrix()
			if m != aU {
				m.checkOverlap(amat)
			}
			if m != bU {
				m.checkOverlap(bmat)
			}
			for ja, jb, jm := 0, 0, 0; ja < ar*amat.Stride; ja, jb, jm = ja+amat.Stride, jb+bmat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v / bmat.Data[i+jb]
				}
			}
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}

	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, a.At(r, c)/b.At(r, c))
		}
	}
}

// Mul takes the matrix product of a and b, placing the result in the receiver.
// If the number of columns in a does not equal the number of rows in b, Mul will panic.
func (m *Dense) Mul(a, b Matrix) {
	ar, ac := a.Dims()
	br, bc := b.Dims()

	if ac != br {
		panic(ErrShape)
	}

	aU, aTrans := untranspose(a)
	bU, bTrans := untranspose(b)
	m.reuseAs(ar, bc)
	var restore func()
	if m == aU {
		m, restore = m.isolatedWorkspace(aU)
		defer restore()
	} else if m == bU {
		m, restore = m.isolatedWorkspace(bU)
		defer restore()
	}
	aT := blas.NoTrans
	if aTrans {
		aT = blas.Trans
	}
	bT := blas.NoTrans
	if bTrans {
		bT = blas.Trans
	}

	// Some of the cases do not have a transpose option, so create
	// temporary memory.
	// C = A^T * B = (B^T * A)^T
	// C^T = B^T * A.
	if aUrm, ok := aU.(RawMatrixer); ok {
		amat := aUrm.RawMatrix()
		if restore == nil {
			m.checkOverlap(amat)
		}
		if bUrm, ok := bU.(RawMatrixer); ok {
			bmat := bUrm.RawMatrix()
			if restore == nil {
				m.checkOverlap(bmat)
			}
			if m.mat.Data == nil {
				panic("blas: index of c out of range")
			}

			blas32.Gemm(aT, bT, 1, amat, bmat, 0, m.mat)
			return
		}

		if bU, ok := bU.(RawTriangular); ok {
			// Trmm updates in place, so copy aU first.
			bmat := bU.RawTriangular()
			if aTrans {
				c := getWorkspace(ac, ar, false)
				var tmp Dense
				tmp.SetRawMatrix(amat)
				c.Copy(&tmp)
				bT := blas.Trans
				if bTrans {
					bT = blas.NoTrans
				}
				blas32.Trmm(blas.Left, bT, 1, bmat, c.mat)
				strictCopy(m, c.T())
				putWorkspace(c)
				return
			}
			m.Copy(a)
			blas32.Trmm(blas.Right, bT, 1, bmat, m.mat)
			return
		}
		if bU, ok := bU.(*VecDense); ok {
			m.checkOverlap(bU.asGeneral())
			bvec := bU.RawVector()
			if bTrans {
				// {ar,1} x {1,bc}, which is not a vector.
				// Instead, construct B as a General.
				bmat := blas32.General{
					Rows:   bc,
					Cols:   1,
					Stride: bvec.Inc,
					Data:   bvec.Data,
				}
				blas32.Gemm(aT, bT, 1, amat, bmat, 0, m.mat)
				return
			}
			cvec := blas32.Vector{
				Inc:  m.mat.Stride,
				Data: m.mat.Data,
			}
			blas32.Gemv(aT, 1, amat, bvec, 0, cvec)
			return
		}
	}
	if bUrm, ok := bU.(RawMatrixer); ok {
		bmat := bUrm.RawMatrix()
		if restore == nil {
			m.checkOverlap(bmat)
		}
		if aU, ok := aU.(RawTriangular); ok {
			// Trmm updates in place, so copy bU first.
			amat := aU.RawTriangular()
			if bTrans {
				c := getWorkspace(bc, br, false)
				var tmp Dense
				tmp.SetRawMatrix(bmat)
				c.Copy(&tmp)
				aT := blas.Trans
				if aTrans {
					aT = blas.NoTrans
				}
				blas32.Trmm(blas.Right, aT, 1, amat, c.mat)
				strictCopy(m, c.T())
				putWorkspace(c)
				return
			}
			m.Copy(b)
			blas32.Trmm(blas.Left, aT, 1, amat, m.mat)
			return
		}
		if aU, ok := aU.(*VecDense); ok {
			m.checkOverlap(aU.asGeneral())
			avec := aU.RawVector()
			if aTrans {
				// {1,ac} x {ac, bc}
				// Transpose B so that the vector is on the right.
				cvec := blas32.Vector{
					Inc:  1,
					Data: m.mat.Data,
				}
				bT := blas.Trans
				if bTrans {
					bT = blas.NoTrans
				}
				blas32.Gemv(bT, 1, bmat, avec, 0, cvec)
				return
			}
			// {ar,1} x {1,bc} which is not a vector result.
			// Instead, construct A as a General.
			amat := blas32.General{
				Rows:   ar,
				Cols:   1,
				Stride: avec.Inc,
				Data:   avec.Data,
			}
			blas32.Gemm(aT, bT, 1, amat, bmat, 0, m.mat)
			return
		}
	}

	m.checkOverlapMatrix(aU)
	m.checkOverlapMatrix(bU)
	row := getFloats(ac, false)
	defer putFloats(row)
	for r := 0; r < ar; r++ {
		for i := range row {
			row[i] = a.At(r, i)
		}
		for c := 0; c < bc; c++ {
			var v float32
			for i, e := range row {
				v += e * b.At(i, c)
			}
			m.mat.Data[r*m.mat.Stride+c] = v
		}
	}
}

// strictCopy copies a into m panicking if the shape of a and m differ.
func strictCopy(m *Dense, a Matrix) {
	r, c := m.Copy(a)
	if r != m.mat.Rows || c != m.mat.Cols {
		// Panic with a string since this
		// is not a user-facing panic.
		panic(ErrShape.Error())
	}
}

// Pow calculates the integral power of the matrix a to n, placing the result
// in the receiver. Pow will panic if n is negative or if a is not square.
func (m *Dense) Pow(a Matrix, n int) {
	if n < 0 {
		panic("matrix: illegal power")
	}
	r, c := a.Dims()
	if r != c {
		panic(ErrShape)
	}

	m.reuseAs(r, c)

	// Take possible fast paths.
	switch n {
	case 0:
		for i := 0; i < r; i++ {
			zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c])
			m.mat.Data[i*m.mat.Stride+i] = 1
		}
		return
	case 1:
		m.Copy(a)
		return
	case 2:
		m.Mul(a, a)
		return
	}

	// Perform iterative exponentiation by squaring in work space.
	w := getWorkspace(r, r, false)
	w.Copy(a)
	s := getWorkspace(r, r, false)
	s.Copy(a)
	x := getWorkspace(r, r, false)
	for n--; n > 0; n >>= 1 {
		if n&1 != 0 {
			x.Mul(w, s)
			w, x = x, w
		}
		if n != 1 {
			x.Mul(s, s)
			s, x = x, s
		}
	}
	m.Copy(w)
	putWorkspace(w)
	putWorkspace(s)
	putWorkspace(x)
}

// Scale multiplies the elements of a by f, placing the result in the receiver.
//
// See the Scaler interface for more information.
func (m *Dense) Scale(f float32, a Matrix) {
	ar, ac := a.Dims()

	m.reuseAs(ar, ac)

	aU, aTrans := untranspose(a)
	if rm, ok := aU.(RawMatrixer); ok {
		amat := rm.RawMatrix()
		if m == aU || m.checkOverlap(amat) {
			var restore func()
			m, restore = m.isolatedWorkspace(a)
			defer restore()
		}
		if !aTrans {
			for ja, jm := 0, 0; ja < ar*amat.Stride; ja, jm = ja+amat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = v * f
				}
			}
		} else {
			for ja, jm := 0, 0; ja < ac*amat.Stride; ja, jm = ja+amat.Stride, jm+1 {
				for i, v := range amat.Data[ja : ja+ar] {
					m.mat.Data[i*m.mat.Stride+jm] = v * f
				}
			}
		}
		return
	}

	m.checkOverlapMatrix(a)
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, f*a.At(r, c))
		}
	}
}

// Apply applies the function fn to each of the elements of a, placing the
// resulting matrix in the receiver. The function fn takes a row/column
// index and element value and returns some function of that tuple.
func (m *Dense) Apply(fn func(i, j int, v float32) float32, a Matrix) {
	ar, ac := a.Dims()

	m.reuseAs(ar, ac)

	aU, aTrans := untranspose(a)
	if rm, ok := aU.(RawMatrixer); ok {
		amat := rm.RawMatrix()
		if m == aU || m.checkOverlap(amat) {
			var restore func()
			m, restore = m.isolatedWorkspace(a)
			defer restore()
		}
		if !aTrans {
			for j, ja, jm := 0, 0, 0; ja < ar*amat.Stride; j, ja, jm = j+1, ja+amat.Stride, jm+m.mat.Stride {
				for i, v := range amat.Data[ja : ja+ac] {
					m.mat.Data[i+jm] = fn(j, i, v)
				}
			}
		} else {
			for j, ja, jm := 0, 0, 0; ja < ac*amat.Stride; j, ja, jm = j+1, ja+amat.Stride, jm+1 {
				for i, v := range amat.Data[ja : ja+ar] {
					m.mat.Data[i*m.mat.Stride+jm] = fn(i, j, v)
				}
			}
		}
		return
	}

	m.checkOverlapMatrix(a)
	for r := 0; r < ar; r++ {
		for c := 0; c < ac; c++ {
			m.set(r, c, fn(r, c, a.At(r, c)))
		}
	}
}

// RankOne performs a rank-one update to the matrix a and stores the result
// in the receiver. If a is zero, see Outer.
//  m = a + alpha * x * y'
func (m *Dense) RankOne(a Matrix, alpha float32, x, y Vector) {
	ar, ac := a.Dims()
	xr, xc := x.Dims()
	if xr != ar || xc != 1 {
		panic(ErrShape)
	}
	yr, yc := y.Dims()
	if yr != ac || yc != 1 {
		panic(ErrShape)
	}

	if a != m {
		aU, _ := untranspose(a)
		if rm, ok := aU.(RawMatrixer); ok {
			m.checkOverlap(rm.RawMatrix())
		}
	}

	var xmat, ymat blas32.Vector
	fast := true
	xU, _ := untranspose(x)
	if rv, ok := xU.(RawVectorer); ok {
		xmat = rv.RawVector()
		m.checkOverlap((&VecDense{mat: xmat, n: x.Len()}).asGeneral())
	} else {
		fast = false
	}
	yU, _ := untranspose(y)
	if rv, ok := yU.(RawVectorer); ok {
		ymat = rv.RawVector()
		m.checkOverlap((&VecDense{mat: ymat, n: y.Len()}).asGeneral())
	} else {
		fast = false
	}

	if fast {
		if m != a {
			m.reuseAs(ar, ac)
			m.Copy(a)
		}
		blas32.Ger(alpha, xmat, ymat, m.mat)
		return
	}

	m.reuseAs(ar, ac)
	for i := 0; i < ar; i++ {
		for j := 0; j < ac; j++ {
			m.set(i, j, a.At(i, j)+alpha*x.AtVec(i)*y.AtVec(j))
		}
	}
}

// Outer calculates the outer product of the column vectors x and y,
// and stores the result in the receiver.
//  m = alpha * x * y'
// In order to update an existing matrix, see RankOne.
func (m *Dense) Outer(alpha float32, x, y Vector) {
	xr, xc := x.Dims()
	if xc != 1 {
		panic(ErrShape)
	}
	yr, yc := y.Dims()
	if yc != 1 {
		panic(ErrShape)
	}

	r := xr
	c := yr

	// Copied from reuseAs with use replaced by useZeroed
	// and a final zero of the matrix elements if we pass
	// the shape checks.
	// TODO(kortschak): Factor out into reuseZeroedAs if
	// we find another case that needs it.
	if m.mat.Rows > m.capRows || m.mat.Cols > m.capCols {
		// Panic as a string, not a mat.Error.
		panic("mat: caps not correctly set")
	}
	if m.IsZero() {
		m.mat = blas32.General{
			Rows:   r,
			Cols:   c,
			Stride: c,
			Data:   useZeroed(m.mat.Data, r*c),
		}
		m.capRows = r
		m.capCols = c
	} else if r != m.mat.Rows || c != m.mat.Cols {
		panic(ErrShape)
	}

	var xmat, ymat blas32.Vector
	fast := true
	xU, _ := untranspose(x)
	if rv, ok := xU.(RawVectorer); ok {
		xmat = rv.RawVector()
		m.checkOverlap((&VecDense{mat: xmat, n: x.Len()}).asGeneral())

	} else {
		fast = false
	}
	yU, _ := untranspose(y)
	if rv, ok := yU.(RawVectorer); ok {
		ymat = rv.RawVector()
		m.checkOverlap((&VecDense{mat: ymat, n: y.Len()}).asGeneral())
	} else {
		fast = false
	}

	if fast {
		for i := 0; i < r; i++ {
			zero(m.mat.Data[i*m.mat.Stride : i*m.mat.Stride+c])
		}
		blas32.Ger(alpha, xmat, ymat, m.mat)
		return
	}

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.set(i, j, alpha*x.AtVec(i)*y.AtVec(j))
		}
	}
}
