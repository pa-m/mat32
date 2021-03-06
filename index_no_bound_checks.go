// Copyright ©2014 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file must be kept in sync with index_bound_checks.go.

//+build !bounds

package mat32

// At returns the element at row i, column j.
func (m *Dense) At(i, j int) float32 {
	if uint(i) >= uint(m.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(m.mat.Cols) {
		panic(ErrColAccess)
	}
	return m.at(i, j)
}

func (m *Dense) at(i, j int) float32 {
	return m.mat.Data[i*m.mat.Stride+j]
}

// Set sets the element at row i, column j to the value v.
func (m *Dense) Set(i, j int, v float32) {
	if uint(i) >= uint(m.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(m.mat.Cols) {
		panic(ErrColAccess)
	}
	m.set(i, j, v)
}

func (m *Dense) set(i, j int, v float32) {
	m.mat.Data[i*m.mat.Stride+j] = v
}

// At returns the element at row i.
// It panics if i is out of bounds or if j is not zero.
func (v *VecDense) At(i, j int) float32 {
	if uint(i) >= uint(v.n) {
		panic(ErrRowAccess)
	}
	if j != 0 {
		panic(ErrColAccess)
	}
	return v.at(i)
}

// AtVec returns the element at row i.
// It panics if i is out of bounds.
func (v *VecDense) AtVec(i int) float32 {
	if uint(i) >= uint(v.n) {
		panic(ErrRowAccess)
	}
	return v.at(i)
}

func (v *VecDense) at(i int) float32 {
	return v.mat.Data[i*v.mat.Inc]
}

// SetVec sets the element at row i to the value val.
// It panics if i is out of bounds.
func (v *VecDense) SetVec(i int, val float32) {
	if uint(i) >= uint(v.n) {
		panic(ErrVectorAccess)
	}
	v.setVec(i, val)
}

func (v *VecDense) setVec(i int, val float32) {
	v.mat.Data[i*v.mat.Inc] = val
}

// At returns the element at row i, column j.
func (t *TriDense) At(i, j int) float32 {
	if uint(i) >= uint(t.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(t.mat.N) {
		panic(ErrColAccess)
	}
	return t.at(i, j)
}

func (t *TriDense) at(i, j int) float32 {
	isUpper := t.triKind()
	if (isUpper && i > j) || (!isUpper && i < j) {
		return 0
	}
	return t.mat.Data[i*t.mat.Stride+j]
}

// SetTri sets the element at row i, column j to the value v.
// It panics if the location is outside the appropriate half of the matrix.
func (t *TriDense) SetTri(i, j int, v float32) {
	if uint(i) >= uint(t.mat.N) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(t.mat.N) {
		panic(ErrColAccess)
	}
	isUpper := t.isUpper()
	if (isUpper && i > j) || (!isUpper && i < j) {
		panic(ErrTriangleSet)
	}
	t.set(i, j, v)
}

func (t *TriDense) set(i, j int, v float32) {
	t.mat.Data[i*t.mat.Stride+j] = v
}

// At returns the element at row i, column j.
func (b *BandDense) At(i, j int) float32 {
	if uint(i) >= uint(b.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(b.mat.Cols) {
		panic(ErrColAccess)
	}
	return b.at(i, j)
}

func (b *BandDense) at(i, j int) float32 {
	pj := j + b.mat.KL - i
	if pj < 0 || b.mat.KL+b.mat.KU+1 <= pj {
		return 0
	}
	return b.mat.Data[i*b.mat.Stride+pj]
}

// SetBand sets the element at row i, column j to the value v.
// It panics if the location is outside the appropriate region of the matrix.
func (b *BandDense) SetBand(i, j int, v float32) {
	if uint(i) >= uint(b.mat.Rows) {
		panic(ErrRowAccess)
	}
	if uint(j) >= uint(b.mat.Cols) {
		panic(ErrColAccess)
	}
	pj := j + b.mat.KL - i
	if pj < 0 || b.mat.KL+b.mat.KU+1 <= pj {
		panic(ErrBandSet)
	}
	b.set(i, j, v)
}

func (b *BandDense) set(i, j int, v float32) {
	pj := j + b.mat.KL - i
	b.mat.Data[i*b.mat.Stride+pj] = v
}
