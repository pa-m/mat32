// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/chewxy/math32"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/blas32"
)

func panics(fn func()) (panicked bool, message string) {
	defer func() {
		r := recover()
		panicked = r != nil
		message = fmt.Sprint(r)
	}()
	fn()
	return
}

func flatten(f [][]float32) (r, c int, d []float32) {
	r = len(f)
	if r == 0 {
		panic("bad test: no row")
	}
	c = len(f[0])
	d = make([]float32, 0, r*c)
	for _, row := range f {
		if len(row) != c {
			panic("bad test: ragged input")
		}
		d = append(d, row...)
	}
	return r, c, d
}

func unflatten(r, c int, d []float32) [][]float32 {
	m := make([][]float32, r)
	for i := 0; i < r; i++ {
		m[i] = d[i*c : (i+1)*c]
	}
	return m
}

// eye returns a new identity matrix of size n×n.
func eye(n int) *Dense {
	d := make([]float32, n*n)
	for i := 0; i < n*n; i += n + 1 {
		d[i] = 1
	}
	return NewDense(n, n, d)
}

func TestCol(t *testing.T) {
	for id, af := range [][][]float32{
		{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		},
		{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
			{10, 11, 12},
		},
		{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
			{9, 10, 11, 12},
		},
	} {
		a := NewDense(flatten(af))
		col := make([]float32, a.mat.Rows)
		for j := range af[0] {
			for i := range col {
				col[i] = float32(i*a.mat.Cols + j + 1)
			}

			if got := Col(nil, j, a); !reflect.DeepEqual(got, col) {
				t.Errorf("test %d: unexpected values returned for dense col %d: got: %v want: %v",
					id, j, got, col)
			}

			got := make([]float32, a.mat.Rows)
			if Col(got, j, a); !reflect.DeepEqual(got, col) {
				t.Errorf("test %d: unexpected values filled for dense col %d: got: %v want: %v",
					id, j, got, col)
			}
		}
	}

	denseComparison := func(a *Dense) interface{} {
		r, c := a.Dims()
		ans := make([][]float32, c)
		for j := range ans {
			ans[j] = make([]float32, r)
			for i := range ans[j] {
				ans[j][i] = a.At(i, j)
			}
		}
		return ans
	}

	f := func(a Matrix) interface{} {
		_, c := a.Dims()
		ans := make([][]float32, c)
		for j := range ans {
			ans[j] = Col(nil, j, a)
		}
		return ans
	}
	testOneInputFunc(t, "Col", f, denseComparison, sameAnswerF32SliceOfSlice, isAnyType, isAnySize)

	f = func(a Matrix) interface{} {
		r, c := a.Dims()
		ans := make([][]float32, c)
		for j := range ans {
			ans[j] = make([]float32, r)
			Col(ans[j], j, a)
		}
		return ans
	}
	testOneInputFunc(t, "Col", f, denseComparison, sameAnswerF32SliceOfSlice, isAnyType, isAnySize)
}

func TestRow(t *testing.T) {
	for id, af := range [][][]float32{
		{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
		},
		{
			{1, 2, 3},
			{4, 5, 6},
			{7, 8, 9},
			{10, 11, 12},
		},
		{
			{1, 2, 3, 4},
			{5, 6, 7, 8},
			{9, 10, 11, 12},
		},
	} {
		a := NewDense(flatten(af))
		for i, row := range af {
			if got := Row(nil, i, a); !reflect.DeepEqual(got, row) {
				t.Errorf("test %d: unexpected values returned for dense row %d: got: %v want: %v",
					id, i, got, row)
			}

			got := make([]float32, len(row))
			if Row(got, i, a); !reflect.DeepEqual(got, row) {
				t.Errorf("test %d: unexpected values filled for dense row %d: got: %v want: %v",
					id, i, got, row)
			}
		}
	}

	denseComparison := func(a *Dense) interface{} {
		r, c := a.Dims()
		ans := make([][]float32, r)
		for i := range ans {
			ans[i] = make([]float32, c)
			for j := range ans[i] {
				ans[i][j] = a.At(i, j)
			}
		}
		return ans
	}

	f := func(a Matrix) interface{} {
		r, _ := a.Dims()
		ans := make([][]float32, r)
		for i := range ans {
			ans[i] = Row(nil, i, a)
		}
		return ans
	}
	testOneInputFunc(t, "Row", f, denseComparison, sameAnswerF32SliceOfSlice, isAnyType, isAnySize)

	f = func(a Matrix) interface{} {
		r, c := a.Dims()
		ans := make([][]float32, r)
		for i := range ans {
			ans[i] = make([]float32, c)
			Row(ans[i], i, a)
		}
		return ans
	}
	testOneInputFunc(t, "Row", f, denseComparison, sameAnswerF32SliceOfSlice, isAnyType, isAnySize)
}

type basicVector struct {
	m []float32
}

func (v *basicVector) AtVec(i int) float32 {
	if i < 0 || i >= v.Len() {
		panic(ErrRowAccess)
	}
	return v.m[i]
}

func (v *basicVector) At(r, c int) float32 {
	if c != 0 {
		panic(ErrColAccess)
	}
	return v.AtVec(r)
}

func (v *basicVector) Dims() (r, c int) {
	return v.Len(), 1
}

func (v *basicVector) T() Matrix {
	return Transpose{v}
}

func (v *basicVector) Len() int {
	return len(v.m)
}

func TestDot(t *testing.T) {
	f := func(a, b Matrix) interface{} {
		return Dot(a.(Vector), b.(Vector))
	}
	denseComparison := func(a, b *Dense) interface{} {
		ra, ca := a.Dims()
		rb, cb := b.Dims()
		if ra != rb || ca != cb {
			panic(ErrShape)
		}
		var sum float32
		for i := 0; i < ra; i++ {
			for j := 0; j < ca; j++ {
				sum += a.At(i, j) * b.At(i, j)
			}
		}
		return sum
	}
	testTwoInputFunc(t, "Dot", f, denseComparison, sameAnswerFloatApproxTol(1e-6), legalTypesVectorVector, legalSizeSameVec)
}

func TestEqual(t *testing.T) {
	f := func(a, b Matrix) interface{} {
		return Equal(a, b)
	}
	denseComparison := func(a, b *Dense) interface{} {
		return Equal(a, b)
	}
	testTwoInputFunc(t, "Equal", f, denseComparison, sameAnswerBool, legalTypesAll, isAnySize2)
}

func TestMax(t *testing.T) {
	// A direct test of Max with *Dense arguments is in TestNewDense.
	f := func(a Matrix) interface{} {
		return Max(a)
	}
	denseComparison := func(a *Dense) interface{} {
		return Max(a)
	}
	testOneInputFunc(t, "Max", f, denseComparison, sameAnswerFloat, isAnyType, isAnySize)
}

func TestMin(t *testing.T) {
	// A direct test of Min with *Dense arguments is in TestNewDense.
	f := func(a Matrix) interface{} {
		return Min(a)
	}
	denseComparison := func(a *Dense) interface{} {
		return Min(a)
	}
	testOneInputFunc(t, "Min", f, denseComparison, sameAnswerFloat, isAnyType, isAnySize)
}

func TestNorm(t *testing.T) {
	for i, test := range []struct {
		a    [][]float32
		ord  float32
		norm float32
	}{
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  1,
			norm: 30,
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  2,
			norm: 25.495097567963924,
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
			ord:  math32.Inf(1),
			norm: 33,
		},
		{
			a:    [][]float32{{1, -2, -2}, {-4, 5, 6}},
			ord:  1,
			norm: 8,
		},
		{
			a:    [][]float32{{1, -2, -2}, {-4, 5, 6}},
			ord:  math32.Inf(1),
			norm: 15,
		},
	} {
		a := NewDense(flatten(test.a))
		if math32.Abs(Norm(a, test.ord)-test.norm) > 1e-6 {
			t.Errorf("Mismatch test %d: %v norm = %f got %f ord:%g", i, test.a, test.norm, Norm(a, test.ord), test.ord)
		}
	}

	for _, test := range []struct {
		name string
		norm float32
	}{
		{"NormOne", 1},
		{"NormTwo", 2},
		{"NormInf", math32.Inf(1)},
	} {
		f := func(a Matrix) interface{} {
			return Norm(a, test.norm)
		}
		denseComparison := func(a *Dense) interface{} {
			return Norm(a, test.norm)
		}
		testOneInputFunc(t, test.name, f, denseComparison, sameAnswerFloatApproxTol(1e-12), isAnyType, isAnySize)
	}
}

func TestNormZero(t *testing.T) {
	for _, a := range []Matrix{
		&Dense{},
		&TriDense{},
		&TriDense{mat: blas32.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit}},
		&VecDense{},
	} {
		for _, norm := range []float32{1, 2, math32.Inf(1)} {
			panicked, message := panics(func() { Norm(a, norm) })
			if !panicked {
				t.Errorf("expected panic for Norm(&%T{}, %v)", a, norm)
			}
			if message != ErrShape.Error() {
				t.Errorf("unexpected panic string for Norm(&%T{}, %v): got:%s want:%s",
					a, norm, message, ErrShape.Error())
			}
		}
	}
}

func TestSum(t *testing.T) {
	f := func(a Matrix) interface{} {
		return Sum(a)
	}
	denseComparison := func(a *Dense) interface{} {
		return Sum(a)
	}
	testOneInputFunc(t, "Sum", f, denseComparison, sameAnswerFloatApproxTol(1e-6), isAnyType, isAnySize)
}

func TestTrace(t *testing.T) {
	for _, test := range []struct {
		a     *Dense
		trace float32
	}{
		{
			a:     NewDense(3, 3, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}),
			trace: 15,
		},
	} {
		trace := Trace(test.a)
		if trace != test.trace {
			t.Errorf("Trace mismatch. Want %v, got %v", test.trace, trace)
		}
	}
	f := func(a Matrix) interface{} {
		return Trace(a)
	}
	denseComparison := func(a *Dense) interface{} {
		return Trace(a)
	}
	testOneInputFunc(t, "Trace", f, denseComparison, sameAnswerFloat, isAnyType, isSquare)
}

func TestDoer(t *testing.T) {
	type MatrixDoer interface {
		Matrix
		NonZeroDoer
		RowNonZeroDoer
		ColNonZeroDoer
	}
	ones := func(n int) []float32 {
		data := make([]float32, n)
		for i := range data {
			data[i] = 1
		}
		return data
	}
	for i, m := range []MatrixDoer{
		NewTriDense(3, Lower, ones(3*3)),
		NewTriDense(3, Upper, ones(3*3)),
		NewBandDense(6, 6, 1, 1, ones(3*6)),
		NewBandDense(6, 10, 1, 1, ones(3*6)),
		NewBandDense(10, 6, 1, 1, ones(7*3)),
	} {
		r, c := m.Dims()

		want := Sum(m)

		// got and fn sum the accessed elements in
		// the Doer that is being operated on.
		// fn also tests that the accessed elements
		// are within the writable areas of the
		// matrix to check that only valid elements
		// are operated on.
		var got float32
		fn := func(i, j int, v float32) {
			got += v
			switch m := m.(type) {
			case MutableTriangular:
				m.SetTri(i, j, v)
			case MutableBanded:
				m.SetBand(i, j, v)
			default:
				panic("bad test: need mutable type")
			}
		}

		panicked, message := panics(func() { m.DoNonZero(fn) })
		if panicked {
			t.Errorf("unexpected panic for Doer test %d: %q", i, message)
			continue
		}
		if got != want {
			t.Errorf("unexpected Doer sum: got:%f want:%f", got, want)
		}

		// Reset got for testing with DoRowNonZero.
		got = 0
		panicked, message = panics(func() {
			for i := 0; i < r; i++ {
				m.DoRowNonZero(i, fn)
			}
		})
		if panicked {
			t.Errorf("unexpected panic for RowDoer test %d: %q", i, message)
			continue
		}
		if got != want {
			t.Errorf("unexpected RowDoer sum: got:%f want:%f", got, want)
		}

		// Reset got for testing with DoColNonZero.
		got = 0
		panicked, message = panics(func() {
			for j := 0; j < c; j++ {
				m.DoColNonZero(j, fn)
			}
		})
		if panicked {
			t.Errorf("unexpected panic for ColDoer test %d: %q", i, message)
			continue
		}
		if got != want {
			t.Errorf("unexpected ColDoer sum: got:%f want:%f", got, want)
		}
	}
}
