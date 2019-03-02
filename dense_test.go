// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"reflect"
	"testing"

	"github.com/chewxy/math32"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/blas/blas32"
)

func asBasicMatrix(d *Dense) Matrix            { return (*basicMatrix)(d) }
func asBasicTriangular(t *TriDense) Triangular { return (*basicTriangular)(t) }

func TestNewDense(t *testing.T) {
	for i, test := range []struct {
		a          []float32
		rows, cols int
		min, max   float32
		fro        float32
		mat        *Dense
	}{
		{
			[]float32{
				0, 0, 0,
				0, 0, 0,
				0, 0, 0,
			},
			3, 3,
			0, 0,
			0,
			&Dense{
				mat: blas32.General{
					Rows: 3, Cols: 3,
					Stride: 3,
					Data:   []float32{0, 0, 0, 0, 0, 0, 0, 0, 0},
				},
				capRows: 3, capCols: 3,
			},
		},
		{
			[]float32{
				1, 1, 1,
				1, 1, 1,
				1, 1, 1,
			},
			3, 3,
			1, 1,
			3,
			&Dense{
				mat: blas32.General{
					Rows: 3, Cols: 3,
					Stride: 3,
					Data:   []float32{1, 1, 1, 1, 1, 1, 1, 1, 1},
				},
				capRows: 3, capCols: 3,
			},
		},
		{
			[]float32{
				1, 0, 0,
				0, 1, 0,
				0, 0, 1,
			},
			3, 3,
			0, 1,
			1.7320508075688772,
			&Dense{
				mat: blas32.General{
					Rows: 3, Cols: 3,
					Stride: 3,
					Data:   []float32{1, 0, 0, 0, 1, 0, 0, 0, 1},
				},
				capRows: 3, capCols: 3,
			},
		},
		{
			[]float32{
				-1, 0, 0,
				0, -1, 0,
				0, 0, -1,
			},
			3, 3,
			-1, 0,
			1.7320508075688772,
			&Dense{
				mat: blas32.General{
					Rows: 3, Cols: 3,
					Stride: 3,
					Data:   []float32{-1, 0, 0, 0, -1, 0, 0, 0, -1},
				},
				capRows: 3, capCols: 3,
			},
		},
		{
			[]float32{
				1, 2, 3,
				4, 5, 6,
			},
			2, 3,
			1, 6,
			9.539392014169458,
			&Dense{
				mat: blas32.General{
					Rows: 2, Cols: 3,
					Stride: 3,
					Data:   []float32{1, 2, 3, 4, 5, 6},
				},
				capRows: 2, capCols: 3,
			},
		},
		{
			[]float32{
				1, 2,
				3, 4,
				5, 6,
			},
			3, 2,
			1, 6,
			9.539392014169458,
			&Dense{
				mat: blas32.General{
					Rows: 3, Cols: 2,
					Stride: 2,
					Data:   []float32{1, 2, 3, 4, 5, 6},
				},
				capRows: 3, capCols: 2,
			},
		},
	} {
		m := NewDense(test.rows, test.cols, test.a)
		rows, cols := m.Dims()
		if rows != test.rows {
			t.Errorf("unexpected number of rows for test %d: got: %d want: %d", i, rows, test.rows)
		}
		if cols != test.cols {
			t.Errorf("unexpected number of cols for test %d: got: %d want: %d", i, cols, test.cols)
		}
		if min := Min(m); min != test.min {
			t.Errorf("unexpected min for test %d: got: %v want: %v", i, min, test.min)
		}
		if max := Max(m); max != test.max {
			t.Errorf("unexpected max for test %d: got: %v want: %v", i, max, test.max)
		}
		if fro := Norm(m, 2); math32.Abs(Norm(m, 2)-test.fro) > 1e-14 {
			t.Errorf("unexpected Frobenius norm for test %d: got: %v want: %v", i, fro, test.fro)
		}
		if !reflect.DeepEqual(m, test.mat) {
			t.Errorf("unexpected matrix for test %d", i)
		}
		if !Equal(m, test.mat) {
			t.Errorf("matrix does not equal expected matrix for test %d", i)
		}
	}
}

func TestAtSet(t *testing.T) {
	for test, af := range [][][]float32{
		{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, // even
		{{1, 2}, {4, 5}, {7, 8}},          // wide
		{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, //skinny
	} {
		m := NewDense(flatten(af))
		rows, cols := m.Dims()
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				if m.At(i, j) != af[i][j] {
					t.Errorf("unexpected value for At(%d, %d) for test %d: got: %v want: %v",
						i, j, test, m.At(i, j), af[i][j])
				}

				v := float32(i * j)
				m.Set(i, j, v)
				if m.At(i, j) != v {
					t.Errorf("unexpected value for At(%d, %d) after Set(%[1]d, %d, %v) for test %d: got: %v want: %[3]v",
						i, j, v, test, m.At(i, j))
				}
			}
		}
		// Check access out of bounds fails
		for _, row := range []int{-1, rows, rows + 1} {
			panicked, message := panics(func() { m.At(row, 0) })
			if !panicked || message != ErrRowAccess.Error() {
				t.Errorf("expected panic for invalid row access N=%d r=%d", rows, row)
			}
		}
		for _, col := range []int{-1, cols, cols + 1} {
			panicked, message := panics(func() { m.At(0, col) })
			if !panicked || message != ErrColAccess.Error() {
				t.Errorf("expected panic for invalid column access N=%d c=%d", cols, col)
			}
		}

		// Check Set out of bounds
		for _, row := range []int{-1, rows, rows + 1} {
			panicked, message := panics(func() { m.Set(row, 0, 1.2) })
			if !panicked || message != ErrRowAccess.Error() {
				t.Errorf("expected panic for invalid row access N=%d r=%d", rows, row)
			}
		}
		for _, col := range []int{-1, cols, cols + 1} {
			panicked, message := panics(func() { m.Set(0, col, 1.2) })
			if !panicked || message != ErrColAccess.Error() {
				t.Errorf("expected panic for invalid column access N=%d c=%d", cols, col)
			}
		}
	}
}

func TestSetRowColumn(t *testing.T) {
	for _, as := range [][][]float32{
		{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}},
		{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
	} {
		for ri, row := range as {
			a := NewDense(flatten(as))
			m := &Dense{}
			m.Clone(a)
			a.SetRow(ri, make([]float32, a.mat.Cols))
			m.Sub(m, a)
			nt := Norm(m, 2)
			nr := floatsNorm(row, 2)
			if math32.Abs(nt-nr) > 1e-14 {
				t.Errorf("Row %d norm mismatch, want: %g, got: %g", ri, nr, nt)
			}
		}

		for ci := range as[0] {
			a := NewDense(flatten(as))
			m := &Dense{}
			m.Clone(a)
			a.SetCol(ci, make([]float32, a.mat.Rows))
			col := make([]float32, a.mat.Rows)
			for j := range col {
				col[j] = float32(ci + 1 + j*a.mat.Cols)
			}
			m.Sub(m, a)
			nt := Norm(m, 2)
			nc := floatsNorm(col, 2)
			if math32.Abs(nt-nc) > 1e-14 {
				t.Errorf("Column %d norm mismatch, want: %g, got: %g", ci, nc, nt)
			}
		}
	}
}

func TestRowColView(t *testing.T) {
	for _, test := range []struct {
		mat [][]float32
	}{
		{
			mat: [][]float32{
				{1, 2, 3, 4, 5},
				{6, 7, 8, 9, 10},
				{11, 12, 13, 14, 15},
				{16, 17, 18, 19, 20},
				{21, 22, 23, 24, 25},
			},
		},
		{
			mat: [][]float32{
				{1, 2, 3, 4},
				{6, 7, 8, 9},
				{11, 12, 13, 14},
				{16, 17, 18, 19},
				{21, 22, 23, 24},
			},
		},
		{
			mat: [][]float32{
				{1, 2, 3, 4, 5},
				{6, 7, 8, 9, 10},
				{11, 12, 13, 14, 15},
				{16, 17, 18, 19, 20},
			},
		},
	} {
		// This over cautious approach to building a matrix data
		// slice is to ensure that changes to flatten in the future
		// do not mask a regression to the issue identified in
		// gonum/matrix#110.
		rows, cols, flat := flatten(test.mat)
		m := NewDense(rows, cols, flat[:len(flat):len(flat)])

		for _, row := range []int{-1, rows, rows + 1} {
			panicked, message := panics(func() { m.At(row, 0) })
			if !panicked || message != ErrRowAccess.Error() {
				t.Errorf("expected panic for invalid row access rows=%d r=%d", rows, row)
			}
		}
		for _, col := range []int{-1, cols, cols + 1} {
			panicked, message := panics(func() { m.At(0, col) })
			if !panicked || message != ErrColAccess.Error() {
				t.Errorf("expected panic for invalid column access cols=%d c=%d", cols, col)
			}
		}

		for i := 0; i < rows; i++ {
			vr := m.RowView(i)
			if vr.Len() != cols {
				t.Errorf("unexpected number of columns: got: %d want: %d", vr.Len(), cols)
			}
			for j := 0; j < cols; j++ {
				if got := vr.At(j, 0); got != test.mat[i][j] {
					t.Errorf("unexpected value for row.At(%d, 0): got: %v want: %v",
						j, got, test.mat[i][j])
				}
			}
		}
		for j := 0; j < cols; j++ {
			vc := m.ColView(j)
			if vc.Len() != rows {
				t.Errorf("unexpected number of rows: got: %d want: %d", vc.Len(), rows)
			}
			for i := 0; i < rows; i++ {
				if got := vc.At(i, 0); got != test.mat[i][j] {
					t.Errorf("unexpected value for col.At(%d, 0): got: %v want: %v",
						i, got, test.mat[i][j])
				}
			}
		}
		m = m.Slice(1, rows-1, 1, cols-1).(*Dense)
		for i := 1; i < rows-1; i++ {
			vr := m.RowView(i - 1)
			if vr.Len() != cols-2 {
				t.Errorf("unexpected number of columns: got: %d want: %d", vr.Len(), cols-2)
			}
			for j := 1; j < cols-1; j++ {
				if got := vr.At(j-1, 0); got != test.mat[i][j] {
					t.Errorf("unexpected value for row.At(%d, 0): got: %v want: %v",
						j-1, got, test.mat[i][j])
				}
			}
		}
		for j := 1; j < cols-1; j++ {
			vc := m.ColView(j - 1)
			if vc.Len() != rows-2 {
				t.Errorf("unexpected number of rows: got: %d want: %d", vc.Len(), rows-2)
			}
			for i := 1; i < rows-1; i++ {
				if got := vc.At(i-1, 0); got != test.mat[i][j] {
					t.Errorf("unexpected value for col.At(%d, 0): got: %v want: %v",
						i-1, got, test.mat[i][j])
				}
			}
		}
	}
}

func TestGrow(t *testing.T) {
	m := &Dense{}
	m = m.Grow(10, 10).(*Dense)
	rows, cols := m.Dims()
	capRows, capCols := m.Caps()
	if rows != 10 {
		t.Errorf("unexpected value for rows: got: %d want: 10", rows)
	}
	if cols != 10 {
		t.Errorf("unexpected value for cols: got: %d want: 10", cols)
	}
	if capRows != 10 {
		t.Errorf("unexpected value for capRows: got: %d want: 10", capRows)
	}
	if capCols != 10 {
		t.Errorf("unexpected value for capCols: got: %d want: 10", capCols)
	}

	// Test grow within caps is in-place.
	m.Set(1, 1, 1)
	v := m.Slice(1, 5, 1, 5).(*Dense)
	if v.At(0, 0) != m.At(1, 1) {
		t.Errorf("unexpected viewed element value: got: %v want: %v", v.At(0, 0), m.At(1, 1))
	}
	v = v.Grow(5, 5).(*Dense)
	if !Equal(v, m.Slice(1, 10, 1, 10)) {
		t.Error("unexpected view value after grow")
	}

	// Test grow bigger than caps copies.
	v = v.Grow(5, 5).(*Dense)
	if !Equal(v.Slice(0, 9, 0, 9), m.Slice(1, 10, 1, 10)) {
		t.Error("unexpected mismatched common view value after grow")
	}
	v.Set(0, 0, 0)
	if Equal(v.Slice(0, 9, 0, 9), m.Slice(1, 10, 1, 10)) {
		t.Error("unexpected matching view value after grow past capacity")
	}

	// Test grow uses existing data slice when matrix is zero size.
	v.Reset()
	p, l := &v.mat.Data[:1][0], cap(v.mat.Data)
	*p = 1 // This element is at position (-1, -1) relative to v and so should not be visible.
	v = v.Grow(5, 5).(*Dense)
	if &v.mat.Data[:1][0] != p {
		t.Error("grow unexpectedly copied slice within cap limit")
	}
	if cap(v.mat.Data) != l {
		t.Errorf("unexpected change in data slice capacity: got: %d want: %d", cap(v.mat.Data), l)
	}
	if v.At(0, 0) != 0 {
		t.Errorf("unexpected value for At(0, 0): got: %v want: 0", v.At(0, 0))
	}
}

func TestAdd(t *testing.T) {
	for i, test := range []struct {
		a, b, r [][]float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}},
		},
		{
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-2, 0, 0}, {0, -2, 0}, {0, 0, -2}},
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{2, 4, 6}, {8, 10, 12}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		var temp Dense
		temp.Add(a, b)
		if !Equal(&temp, r) {
			t.Errorf("unexpected result from Add for test %d %v Add %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		zero(temp.mat.Data)
		temp.Add(a, b)
		if !Equal(&temp, r) {
			t.Errorf("unexpected result from Add for test %d %v Add %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		panicked, message := panics(func() { temp.Add(a, b) })
		if !panicked || message != "runtime error: index out of range" {
			t.Error("exected runtime panic for nil data slice")
		}

		a.Add(a, b)
		if !Equal(a, r) {
			t.Errorf("unexpected result from Add for test %d %v Add %v: got: %v want: %v",
				i, test.a, test.b, unflatten(a.mat.Rows, a.mat.Cols, a.mat.Data), test.r)
		}
	}

	panicked, message := panics(func() {
		m := NewDense(10, 10, nil)
		a := NewDense(5, 5, nil)
		m.Slice(1, 6, 1, 6).(*Dense).Add(a, m.Slice(2, 7, 2, 7))
	})
	if !panicked {
		t.Error("expected panic for overlapping matrices")
	}
	if message != regionOverlap {
		t.Errorf("unexpected panic message: got: %q want: %q", message, regionOverlap)
	}

	method := func(receiver, a, b Matrix) {
		type Adder interface {
			Add(a, b Matrix)
		}
		rd := receiver.(Adder)
		rd.Add(a, b)
	}
	denseComparison := func(receiver, a, b *Dense) {
		receiver.Add(a, b)
	}
	testTwoInput(t, "Add", &Dense{}, method, denseComparison, legalTypesAll, legalSizeSameRectangular, 1e-7)
}

func TestSub(t *testing.T) {
	for i, test := range []struct {
		a, b, r [][]float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{0, 0, 0}, {0, 0, 0}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		var temp Dense
		temp.Sub(a, b)
		if !Equal(&temp, r) {
			t.Errorf("unexpected result from Sub for test %d %v Sub %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		zero(temp.mat.Data)
		temp.Sub(a, b)
		if !Equal(&temp, r) {
			t.Errorf("unexpected result from Sub for test %d %v Sub %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		panicked, message := panics(func() { temp.Sub(a, b) })
		if !panicked || message != "runtime error: index out of range" {
			t.Error("exected runtime panic for nil data slice")
		}

		a.Sub(a, b)
		if !Equal(a, r) {
			t.Errorf("unexpected result from Sub for test %d %v Sub %v: got: %v want: %v",
				i, test.a, test.b, unflatten(a.mat.Rows, a.mat.Cols, a.mat.Data), test.r)
		}
	}

	panicked, message := panics(func() {
		m := NewDense(10, 10, nil)
		a := NewDense(5, 5, nil)
		m.Slice(1, 6, 1, 6).(*Dense).Sub(a, m.Slice(2, 7, 2, 7))
	})
	if !panicked {
		t.Error("expected panic for overlapping matrices")
	}
	if message != regionOverlap {
		t.Errorf("unexpected panic message: got: %q want: %q", message, regionOverlap)
	}

	method := func(receiver, a, b Matrix) {
		type Suber interface {
			Sub(a, b Matrix)
		}
		rd := receiver.(Suber)
		rd.Sub(a, b)
	}
	denseComparison := func(receiver, a, b *Dense) {
		receiver.Sub(a, b)
	}
	testTwoInput(t, "Sub", &Dense{}, method, denseComparison, legalTypesAll, legalSizeSameRectangular, 1e-7)
}

func TestMulElem(t *testing.T) {
	for i, test := range []struct {
		a, b, r [][]float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 4, 9}, {16, 25, 36}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		var temp Dense
		temp.MulElem(a, b)
		if !Equal(&temp, r) {
			t.Errorf("unexpected result from MulElem for test %d %v MulElem %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		zero(temp.mat.Data)
		temp.MulElem(a, b)
		if !Equal(&temp, r) {
			t.Errorf("unexpected result from MulElem for test %d %v MulElem %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		panicked, message := panics(func() { temp.MulElem(a, b) })
		if !panicked || message != "runtime error: index out of range" {
			t.Error("exected runtime panic for nil data slice")
		}

		a.MulElem(a, b)
		if !Equal(a, r) {
			t.Errorf("unexpected result from MulElem for test %d %v MulElem %v: got: %v want: %v",
				i, test.a, test.b, unflatten(a.mat.Rows, a.mat.Cols, a.mat.Data), test.r)
		}
	}

	panicked, message := panics(func() {
		m := NewDense(10, 10, nil)
		a := NewDense(5, 5, nil)
		m.Slice(1, 6, 1, 6).(*Dense).MulElem(a, m.Slice(2, 7, 2, 7))
	})
	if !panicked {
		t.Error("expected panic for overlapping matrices")
	}
	if message != regionOverlap {
		t.Errorf("unexpected panic message: got: %q want: %q", message, regionOverlap)
	}

	method := func(receiver, a, b Matrix) {
		type ElemMuler interface {
			MulElem(a, b Matrix)
		}
		rd := receiver.(ElemMuler)
		rd.MulElem(a, b)
	}
	denseComparison := func(receiver, a, b *Dense) {
		receiver.MulElem(a, b)
	}
	testTwoInput(t, "MulElem", &Dense{}, method, denseComparison, legalTypesAll, legalSizeSameRectangular, 1e-7)
}

// A comparison that treats NaNs as equal, for testing.
func (m *Dense) same(b Matrix) bool {
	br, bc := b.Dims()
	if br != m.mat.Rows || bc != m.mat.Cols {
		return false
	}
	for r := 0; r < br; r++ {
		for c := 0; c < bc; c++ {
			if av, bv := m.At(r, c), b.At(r, c); av != bv && !(math32.IsNaN(av) && math32.IsNaN(bv)) {
				return false
			}
		}
	}
	return true
}

func TestDivElem(t *testing.T) {
	for i, test := range []struct {
		a, b, r [][]float32
	}{
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{math32.Inf(1), math32.NaN(), math32.NaN()}, {math32.NaN(), math32.Inf(1), math32.NaN()}, {math32.NaN(), math32.NaN(), math32.Inf(1)}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, math32.NaN(), math32.NaN()}, {math32.NaN(), 1, math32.NaN()}, {math32.NaN(), math32.NaN(), 1}},
		},
		{
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{1, math32.NaN(), math32.NaN()}, {math32.NaN(), 1, math32.NaN()}, {math32.NaN(), math32.NaN(), 1}},
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 1, 1}, {1, 1, 1}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		var temp Dense
		temp.DivElem(a, b)
		if !temp.same(r) {
			t.Errorf("unexpected result from DivElem for test %d %v DivElem %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		zero(temp.mat.Data)
		temp.DivElem(a, b)
		if !temp.same(r) {
			t.Errorf("unexpected result from DivElem for test %d %v DivElem %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		panicked, message := panics(func() { temp.DivElem(a, b) })
		if !panicked || message != "runtime error: index out of range" {
			t.Error("exected runtime panic for nil data slice")
		}

		a.DivElem(a, b)
		if !a.same(r) {
			t.Errorf("unexpected result from DivElem for test %d %v DivElem %v: got: %v want: %v",
				i, test.a, test.b, unflatten(a.mat.Rows, a.mat.Cols, a.mat.Data), test.r)
		}
	}

	panicked, message := panics(func() {
		m := NewDense(10, 10, nil)
		a := NewDense(5, 5, nil)
		m.Slice(1, 6, 1, 6).(*Dense).DivElem(a, m.Slice(2, 7, 2, 7))
	})
	if !panicked {
		t.Error("expected panic for overlapping matrices")
	}
	if message != regionOverlap {
		t.Errorf("unexpected panic message: got: %q want: %q", message, regionOverlap)
	}

	method := func(receiver, a, b Matrix) {
		type ElemDiver interface {
			DivElem(a, b Matrix)
		}
		rd := receiver.(ElemDiver)
		rd.DivElem(a, b)
	}
	denseComparison := func(receiver, a, b *Dense) {
		receiver.DivElem(a, b)
	}
	testTwoInput(t, "DivElem", &Dense{}, method, denseComparison, legalTypesAll, legalSizeSameRectangular, 1e-7)
}

func TestMul(t *testing.T) {
	for i, test := range []struct {
		a, b, r [][]float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{3, 3, 3}, {3, 3, 3}, {3, 3, 3}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 2}, {3, 4}, {5, 6}},
			[][]float32{{22, 28}, {49, 64}},
		},
		{
			[][]float32{{0, 1, 1}, {0, 1, 1}, {0, 1, 1}},
			[][]float32{{0, 1, 1}, {0, 1, 1}, {0, 1, 1}},
			[][]float32{{0, 2, 2}, {0, 2, 2}, {0, 2, 2}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))
		r := NewDense(flatten(test.r))

		var temp Dense
		temp.Mul(a, b)
		if !Equal(&temp, r) {
			t.Errorf("unexpected result from Mul for test %d %v Mul %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		zero(temp.mat.Data)
		temp.Mul(a, b)
		if !Equal(&temp, r) {
			t.Errorf("unexpected result from Mul for test %d %v Mul %v: got: %v want: %v",
				i, test.a, test.b, unflatten(temp.mat.Rows, temp.mat.Cols, temp.mat.Data), test.r)
		}

		// These probably warrant a better check and failure. They should never happen in the wild though.
		temp.mat.Data = nil
		panicked, message := panics(func() { temp.Mul(a, b) })
		if !panicked || message != "blas: index of c out of range" {
			if message != "" {
				t.Errorf("expected runtime panic for nil data slice: got %q", message)
			} else {
				t.Error("expected runtime panic for nil data slice")
			}
		}
	}

	panicked, message := panics(func() {
		m := NewDense(10, 10, nil)
		a := NewDense(5, 5, nil)
		m.Slice(1, 6, 1, 6).(*Dense).Mul(a, m.Slice(2, 7, 2, 7))
	})
	if !panicked {
		t.Error("expected panic for overlapping matrices")
	}
	if message != regionOverlap {
		t.Errorf("unexpected panic message: got: %q want: %q", message, regionOverlap)
	}

	method := func(receiver, a, b Matrix) {
		type Muler interface {
			Mul(a, b Matrix)
		}
		rd := receiver.(Muler)
		rd.Mul(a, b)
	}
	denseComparison := func(receiver, a, b *Dense) {
		receiver.Mul(a, b)
	}
	legalSizeMul := func(ar, ac, br, bc int) bool {
		return ac == br
	}
	testTwoInput(t, "Mul", &Dense{}, method, denseComparison, legalTypesAll, legalSizeMul, 1e-5)
}

func randDense(size int, rho float32, rnd func() float32) (*Dense, error) {
	if size == 0 {
		return nil, ErrZeroLength
	}
	d := &Dense{
		mat: blas32.General{
			Rows: size, Cols: size, Stride: size,
			Data: make([]float32, size*size),
		},
		capRows: size, capCols: size,
	}
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			if rand.Float32() < rho {
				d.Set(i, j, rnd())
			}
		}
	}
	return d, nil
}

func TestPow(t *testing.T) {
	for i, test := range []struct {
		a    [][]float32
		n    int
		mod  func(*Dense)
		want [][]float32
	}{
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			n:    0,
			want: [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			n:    0,
			want: [][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			mod: func(a *Dense) {
				d := make([]float32, 100)
				for i := range d {
					d[i] = math32.NaN()
				}
				*a = *NewDense(10, 10, d).Slice(1, 4, 1, 4).(*Dense)
			},
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			n:    1,
			want: [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			n:    1,
			want: [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			mod: func(a *Dense) {
				d := make([]float32, 100)
				for i := range d {
					d[i] = math32.NaN()
				}
				*a = *NewDense(10, 10, d).Slice(1, 4, 1, 4).(*Dense)
			},
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			n:    2,
			want: [][]float32{{30, 36, 42}, {66, 81, 96}, {102, 126, 150}},
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			n:    2,
			want: [][]float32{{30, 36, 42}, {66, 81, 96}, {102, 126, 150}},
			mod: func(a *Dense) {
				d := make([]float32, 100)
				for i := range d {
					d[i] = math32.NaN()
				}
				*a = *NewDense(10, 10, d).Slice(1, 4, 1, 4).(*Dense)
			},
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			n:    3,
			want: [][]float32{{468, 576, 684}, {1062, 1305, 1548}, {1656, 2034, 2412}},
		},
		{
			a:    [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			n:    3,
			want: [][]float32{{468, 576, 684}, {1062, 1305, 1548}, {1656, 2034, 2412}},
			mod: func(a *Dense) {
				d := make([]float32, 100)
				for i := range d {
					d[i] = math32.NaN()
				}
				*a = *NewDense(10, 10, d).Slice(1, 4, 1, 4).(*Dense)
			},
		},
	} {
		var got Dense
		if test.mod != nil {
			test.mod(&got)
		}
		got.Pow(NewDense(flatten(test.a)), test.n)
		if !EqualApprox(&got, NewDense(flatten(test.want)), 1e-12) {
			t.Errorf("unexpected result for Pow test %d", i)
		}
	}
}

func TestScale(t *testing.T) {
	for _, f := range []float32{0.5, 1, 3} {
		method := func(receiver, a Matrix) {
			type Scaler interface {
				Scale(f float32, a Matrix)
			}
			rd := receiver.(Scaler)
			rd.Scale(f, a)
		}
		denseComparison := func(receiver, a *Dense) {
			receiver.Scale(f, a)
		}
		testOneInput(t, "Scale", &Dense{}, method, denseComparison, isAnyType, isAnySize, 1e-14)
	}
}

func TestPowN(t *testing.T) {
	for i, test := range []struct {
		a   [][]float32
		mod func(*Dense)
	}{
		{
			a: [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
		},
		{
			a: [][]float32{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			mod: func(a *Dense) {
				d := make([]float32, 100)
				for i := range d {
					d[i] = math32.NaN()
				}
				*a = *NewDense(10, 10, d).Slice(1, 4, 1, 4).(*Dense)
			},
		},
	} {
		for n := 1; n <= 6; n++ {
			var got, want Dense
			if test.mod != nil {
				test.mod(&got)
			}
			got.Pow(NewDense(flatten(test.a)), n)
			want.iterativePow(NewDense(flatten(test.a)), n)
			if !Equal(&got, &want) {
				t.Errorf("unexpected result for iterative Pow test %d", i)
			}
		}
	}
}

func (m *Dense) iterativePow(a Matrix, n int) {
	m.Clone(a)
	for i := 1; i < n; i++ {
		m.Mul(m, a)
	}
}

func TestCloneT(t *testing.T) {
	for i, test := range []struct {
		a, want [][]float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 4}, {2, 5}, {3, 6}},
		},
	} {
		a := NewDense(flatten(test.a))
		want := NewDense(flatten(test.want))

		var got, gotT Dense

		for j := 0; j < 2; j++ {
			got.Clone(a.T())
			if !Equal(&got, want) {
				t.Errorf("expected transpose for test %d iteration %d: %v transpose = %v",
					i, j, test.a, test.want)
			}
			gotT.Clone(got.T())
			if !Equal(&gotT, a) {
				t.Errorf("expected transpose for test %d iteration %d: %v transpose = %v",
					i, j, test.a, test.want)
			}

			zero(got.mat.Data)
		}
	}
}

func TestCopyT(t *testing.T) {
	for i, test := range []struct {
		a, want [][]float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		},
		{
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 4}, {2, 5}, {3, 6}},
		},
	} {
		a := NewDense(flatten(test.a))
		want := NewDense(flatten(test.want))

		ar, ac := a.Dims()
		got := NewDense(ac, ar, nil)
		rr := NewDense(ar, ac, nil)

		for j := 0; j < 2; j++ {
			got.Copy(a.T())
			if !Equal(got, want) {
				t.Errorf("expected transpose for test %d iteration %d: %v transpose = %v",
					i, j, test.a, test.want)
			}
			rr.Copy(got.T())
			if !Equal(rr, a) {
				t.Errorf("expected transpose for test %d iteration %d: %v transpose = %v",
					i, j, test.a, test.want)
			}

			zero(got.mat.Data)
		}
	}
}

func TestCopyDenseAlias(t *testing.T) {
	for _, trans := range []bool{false, true} {
		for di := 0; di < 2; di++ {
			for dj := 0; dj < 2; dj++ {
				for si := 0; si < 2; si++ {
					for sj := 0; sj < 2; sj++ {
						a := NewDense(3, 3, []float32{
							1, 2, 3,
							4, 5, 6,
							7, 8, 9,
						})
						src := a.Slice(si, si+2, sj, sj+2)
						want := DenseCopyOf(src)
						got := a.Slice(di, di+2, dj, dj+2).(*Dense)

						if trans {
							panicked, _ := panics(func() { got.Copy(src.T()) })
							if !panicked {
								t.Errorf("expected panic for transpose aliased copy with offsets dst(%d,%d) src(%d,%d):\ngot:\n%v\nwant:\n%v",
									di, dj, si, sj, Formatted(got), Formatted(want),
								)
							}
							continue
						}

						got.Copy(src)
						if !Equal(got, want) {
							t.Errorf("unexpected aliased copy result with offsets dst(%d,%d) src(%d,%d):\ngot:\n%v\nwant:\n%v",
								di, dj, si, sj, Formatted(got), Formatted(want),
							)
						}
					}
				}
			}
		}
	}
}

func TestCopyVecDenseAlias(t *testing.T) {
	for _, horiz := range []bool{false, true} {
		for do := 0; do < 2; do++ {
			for di := 0; di < 3; di++ {
				for si := 0; si < 3; si++ {
					a := NewDense(3, 3, []float32{
						1, 2, 3,
						4, 5, 6,
						7, 8, 9,
					})
					var src Vector
					var want *Dense
					if horiz {
						src = a.RowView(si)
						want = DenseCopyOf(a.Slice(si, si+1, 0, 2))
					} else {
						src = a.ColView(si)
						want = DenseCopyOf(a.Slice(0, 2, si, si+1))
					}

					var got *Dense
					if horiz {
						got = a.Slice(di, di+1, do, do+2).(*Dense)
						got.Copy(src.T())
					} else {
						got = a.Slice(do, do+2, di, di+1).(*Dense)
						got.Copy(src)
					}

					if !Equal(got, want) {
						t.Errorf("unexpected aliased copy result with offsets dst(%d) src(%d):\ngot:\n%v\nwant:\n%v",
							di, si, Formatted(got), Formatted(want),
						)
					}
				}
			}
		}
	}
}

func identity(r, c int, v float32) float32 { return v }

func TestApply(t *testing.T) {
	for i, test := range []struct {
		a, want [][]float32
		fn      func(r, c int, v float32) float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			identity,
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			identity,
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			identity,
		},
		{
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			[][]float32{{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}},
			identity,
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			identity,
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{2, 4, 6}, {8, 10, 12}},
			func(r, c int, v float32) float32 { return v * 2 },
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{0, 2, 0}, {0, 5, 0}},
			func(r, c int, v float32) float32 {
				if c == 1 {
					return v
				}
				return 0
			},
		},
		{
			[][]float32{{1, 2, 3}, {4, 5, 6}},
			[][]float32{{0, 0, 0}, {4, 5, 6}},
			func(r, c int, v float32) float32 {
				if r == 1 {
					return v
				}
				return 0
			},
		},
	} {
		a := NewDense(flatten(test.a))
		want := NewDense(flatten(test.want))

		var got Dense

		for j := 0; j < 2; j++ {
			got.Apply(test.fn, a)
			if !Equal(&got, want) {
				t.Errorf("unexpected result for test %d iteration %d: got: %v want: %v", i, j, got.mat.Data, want.mat.Data)
			}
		}
	}

	for _, fn := range []func(r, c int, v float32) float32{
		identity,
		func(r, c int, v float32) float32 {
			if r < c {
				return v
			}
			return -v
		},
		func(r, c int, v float32) float32 {
			if r%2 == 0 && c%2 == 0 {
				return v
			}
			return -v
		},
		func(_, _ int, v float32) float32 { return v * v },
		func(_, _ int, v float32) float32 { return -v },
	} {
		method := func(receiver, x Matrix) {
			type Applier interface {
				Apply(func(r, c int, v float32) float32, Matrix)
			}
			rd := receiver.(Applier)
			rd.Apply(fn, x)
		}
		denseComparison := func(receiver, x *Dense) {
			receiver.Apply(fn, x)
		}
		testOneInput(t, "Apply", &Dense{}, method, denseComparison, isAnyType, isAnySize, 0)
	}
}

func TestClone(t *testing.T) {
	for i, test := range []struct {
		a    [][]float32
		i, j int
		v    float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			1, 1,
			1,
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			0, 0,
			0,
		},
	} {
		a := NewDense(flatten(test.a))
		b := *a
		a.Clone(a)
		a.Set(test.i, test.j, test.v)

		if Equal(&b, a) {
			t.Errorf("unexpected mirror of write to cloned matrix for test %d: %v cloned and altered = %v",
				i, a, &b)
		}
	}
}

// TODO(kortschak) Roll this into testOneInput when it exists.
func TestCopyPanic(t *testing.T) {
	for _, a := range []*Dense{
		{},
		{mat: blas32.General{Rows: 1}},
		{mat: blas32.General{Cols: 1}},
	} {
		var rows, cols int
		m := NewDense(1, 1, nil)
		panicked, message := panics(func() { rows, cols = m.Copy(a) })
		if panicked {
			t.Errorf("unexpected panic: %v", message)
		}
		if rows != 0 {
			t.Errorf("unexpected rows: got: %d want: 0", rows)
		}
		if cols != 0 {
			t.Errorf("unexpected cols: got: %d want: 0", cols)
		}
	}
}

func TestStack(t *testing.T) {
	for i, test := range []struct {
		a, b, e [][]float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))

		var s Dense
		s.Stack(a, b)

		if !Equal(&s, NewDense(flatten(test.e))) {
			t.Errorf("unexpected result for Stack test %d: %v stack %v = %v", i, a, b, s)
		}
	}

	method := func(receiver, a, b Matrix) {
		type Stacker interface {
			Stack(a, b Matrix)
		}
		rd := receiver.(Stacker)
		rd.Stack(a, b)
	}
	denseComparison := func(receiver, a, b *Dense) {
		receiver.Stack(a, b)
	}
	testTwoInput(t, "Stack", &Dense{}, method, denseComparison, legalTypesAll, legalSizeSameWidth, 0)
}

func TestAugment(t *testing.T) {
	for i, test := range []struct {
		a, b, e [][]float32
	}{
		{
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}},
			[][]float32{{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}},
		},
		{
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}},
			[][]float32{{1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}},
		},
		{
			[][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			[][]float32{{0, 1, 0}, {0, 0, 1}, {1, 0, 0}},
			[][]float32{{1, 0, 0, 0, 1, 0}, {0, 1, 0, 0, 0, 1}, {0, 0, 1, 1, 0, 0}},
		},
	} {
		a := NewDense(flatten(test.a))
		b := NewDense(flatten(test.b))

		var s Dense
		s.Augment(a, b)

		if !Equal(&s, NewDense(flatten(test.e))) {
			t.Errorf("unexpected result for Augment test %d: %v augment %v = %v", i, a, b, s)
		}
	}

	method := func(receiver, a, b Matrix) {
		type Augmenter interface {
			Augment(a, b Matrix)
		}
		rd := receiver.(Augmenter)
		rd.Augment(a, b)
	}
	denseComparison := func(receiver, a, b *Dense) {
		receiver.Augment(a, b)
	}
	testTwoInput(t, "Augment", &Dense{}, method, denseComparison, legalTypesAll, legalSizeSameHeight, 0)
}

func TestRankOne(t *testing.T) {
	for i, test := range []struct {
		x     []float32
		y     []float32
		m     [][]float32
		alpha float32
	}{
		{
			x:     []float32{5},
			y:     []float32{10},
			m:     [][]float32{{2}},
			alpha: -3,
		},
		{
			x:     []float32{5, 6, 1},
			y:     []float32{10},
			m:     [][]float32{{2}, {-3}, {5}},
			alpha: -3,
		},

		{
			x:     []float32{5},
			y:     []float32{10, 15, 8},
			m:     [][]float32{{2, -3, 5}},
			alpha: -3,
		},
		{
			x: []float32{1, 5},
			y: []float32{10, 15},
			m: [][]float32{
				{2, -3},
				{4, -1},
			},
			alpha: -3,
		},
		{
			x: []float32{2, 3, 9},
			y: []float32{8, 9},
			m: [][]float32{
				{2, 3},
				{4, 5},
				{6, 7},
			},
			alpha: -3,
		},
		{
			x: []float32{2, 3},
			y: []float32{8, 9, 9},
			m: [][]float32{
				{2, 3, 6},
				{4, 5, 7},
			},
			alpha: -3,
		},
	} {
		want := &Dense{}
		xm := NewDense(len(test.x), 1, test.x)
		ym := NewDense(1, len(test.y), test.y)

		want.Mul(xm, ym)
		want.Scale(test.alpha, want)
		want.Add(want, NewDense(flatten(test.m)))

		a := NewDense(flatten(test.m))
		m := &Dense{}
		// Check with a new matrix
		m.RankOne(a, test.alpha, NewVecDense(len(test.x), test.x), NewVecDense(len(test.y), test.y))
		if !Equal(m, want) {
			t.Errorf("unexpected result for RankOne test %d iteration 0: got: %+v want: %+v", i, m, want)
		}
		// Check with the same matrix
		a.RankOne(a, test.alpha, NewVecDense(len(test.x), test.x), NewVecDense(len(test.y), test.y))
		if !Equal(a, want) {
			t.Errorf("unexpected result for RankOne test %d iteration 1: got: %+v want: %+v", i, m, want)
		}
	}
}

func TestOuter(t *testing.T) {
	for i, test := range []struct {
		x []float32
		y []float32
	}{
		{
			x: []float32{5},
			y: []float32{10},
		},
		{
			x: []float32{5, 6, 1},
			y: []float32{10},
		},

		{
			x: []float32{5},
			y: []float32{10, 15, 8},
		},
		{
			x: []float32{1, 5},
			y: []float32{10, 15},
		},
		{
			x: []float32{2, 3, 9},
			y: []float32{8, 9},
		},
		{
			x: []float32{2, 3},
			y: []float32{8, 9, 9},
		},
	} {
		for _, f := range []float32{0.5, 1, 3} {
			want := &Dense{}
			xm := NewDense(len(test.x), 1, test.x)
			ym := NewDense(1, len(test.y), test.y)

			want.Mul(xm, ym)
			want.Scale(f, want)

			var m Dense
			for j := 0; j < 2; j++ {
				// Check with a new matrix - and then again.
				m.Outer(f, NewVecDense(len(test.x), test.x), NewVecDense(len(test.y), test.y))
				if !Equal(&m, want) {
					t.Errorf("unexpected result for Outer test %d iteration %d scale %v: got: %+v want: %+v", i, j, f, m, want)
				}
			}
		}
	}

	for _, alpha := range []float32{0, 1, -1, 2.3, -2.3} {
		method := func(receiver, x, y Matrix) {
			type outerer interface {
				Outer(alpha float32, x, y Vector)
			}
			m := receiver.(outerer)
			m.Outer(alpha, x.(Vector), y.(Vector))
		}
		denseComparison := func(receiver, x, y *Dense) {
			receiver.Mul(x, y.T())
			receiver.Scale(alpha, receiver)
		}
		testTwoInput(t, "Outer", &Dense{}, method, denseComparison, legalTypesVectorVector, legalSizeVector, 1e-6)
	}
}

var (
	wd *Dense
)

func BenchmarkMulDense100Half(b *testing.B)        { denseMulBench(b, 100, 0.5) }
func BenchmarkMulDense100Tenth(b *testing.B)       { denseMulBench(b, 100, 0.1) }
func BenchmarkMulDense1000Half(b *testing.B)       { denseMulBench(b, 1000, 0.5) }
func BenchmarkMulDense1000Tenth(b *testing.B)      { denseMulBench(b, 1000, 0.1) }
func BenchmarkMulDense1000Hundredth(b *testing.B)  { denseMulBench(b, 1000, 0.01) }
func BenchmarkMulDense1000Thousandth(b *testing.B) { denseMulBench(b, 1000, 0.001) }
func denseMulBench(b *testing.B, size int, rho float32) {
	b.StopTimer()
	a, _ := randDense(size, rho, randNormFloat32)
	d, _ := randDense(size, rho, randNormFloat32)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		var n Dense
		n.Mul(a, d)
		wd = &n
	}
}

func BenchmarkPreMulDense100Half(b *testing.B)        { densePreMulBench(b, 100, 0.5) }
func BenchmarkPreMulDense100Tenth(b *testing.B)       { densePreMulBench(b, 100, 0.1) }
func BenchmarkPreMulDense1000Half(b *testing.B)       { densePreMulBench(b, 1000, 0.5) }
func BenchmarkPreMulDense1000Tenth(b *testing.B)      { densePreMulBench(b, 1000, 0.1) }
func BenchmarkPreMulDense1000Hundredth(b *testing.B)  { densePreMulBench(b, 1000, 0.01) }
func BenchmarkPreMulDense1000Thousandth(b *testing.B) { densePreMulBench(b, 1000, 0.001) }
func densePreMulBench(b *testing.B, size int, rho float32) {
	b.StopTimer()
	a, _ := randDense(size, rho, randNormFloat32)
	d, _ := randDense(size, rho, randNormFloat32)
	wd = NewDense(size, size, nil)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		wd.Mul(a, d)
	}
}

func BenchmarkRow10(b *testing.B)   { rowBench(b, 10) }
func BenchmarkRow100(b *testing.B)  { rowBench(b, 100) }
func BenchmarkRow1000(b *testing.B) { rowBench(b, 1000) }

func rowBench(b *testing.B, size int) {
	a, _ := randDense(size, 1, randNormFloat32)
	_, c := a.Dims()
	dst := make([]float32, c)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Row(dst, 0, a)
	}
}

func BenchmarkPow10_3(b *testing.B)   { powBench(b, 10, 3) }
func BenchmarkPow100_3(b *testing.B)  { powBench(b, 100, 3) }
func BenchmarkPow1000_3(b *testing.B) { powBench(b, 1000, 3) }
func BenchmarkPow10_4(b *testing.B)   { powBench(b, 10, 4) }
func BenchmarkPow100_4(b *testing.B)  { powBench(b, 100, 4) }
func BenchmarkPow1000_4(b *testing.B) { powBench(b, 1000, 4) }
func BenchmarkPow10_5(b *testing.B)   { powBench(b, 10, 5) }
func BenchmarkPow100_5(b *testing.B)  { powBench(b, 100, 5) }
func BenchmarkPow1000_5(b *testing.B) { powBench(b, 1000, 5) }
func BenchmarkPow10_6(b *testing.B)   { powBench(b, 10, 6) }
func BenchmarkPow100_6(b *testing.B)  { powBench(b, 100, 6) }
func BenchmarkPow1000_6(b *testing.B) { powBench(b, 1000, 6) }
func BenchmarkPow10_7(b *testing.B)   { powBench(b, 10, 7) }
func BenchmarkPow100_7(b *testing.B)  { powBench(b, 100, 7) }
func BenchmarkPow1000_7(b *testing.B) { powBench(b, 1000, 7) }
func BenchmarkPow10_8(b *testing.B)   { powBench(b, 10, 8) }
func BenchmarkPow100_8(b *testing.B)  { powBench(b, 100, 8) }
func BenchmarkPow1000_8(b *testing.B) { powBench(b, 1000, 8) }
func BenchmarkPow10_9(b *testing.B)   { powBench(b, 10, 9) }
func BenchmarkPow100_9(b *testing.B)  { powBench(b, 100, 9) }
func BenchmarkPow1000_9(b *testing.B) { powBench(b, 1000, 9) }

func powBench(b *testing.B, size, n int) {
	a, _ := randDense(size, 1, randNormFloat32)

	b.ResetTimer()
	var m Dense
	for i := 0; i < b.N; i++ {
		m.Pow(a, n)
	}
}

func BenchmarkMulTransDense100Half(b *testing.B)        { denseMulTransBench(b, 100, 0.5) }
func BenchmarkMulTransDense100Tenth(b *testing.B)       { denseMulTransBench(b, 100, 0.1) }
func BenchmarkMulTransDense1000Half(b *testing.B)       { denseMulTransBench(b, 1000, 0.5) }
func BenchmarkMulTransDense1000Tenth(b *testing.B)      { denseMulTransBench(b, 1000, 0.1) }
func BenchmarkMulTransDense1000Hundredth(b *testing.B)  { denseMulTransBench(b, 1000, 0.01) }
func BenchmarkMulTransDense1000Thousandth(b *testing.B) { denseMulTransBench(b, 1000, 0.001) }
func denseMulTransBench(b *testing.B, size int, rho float32) {
	b.StopTimer()
	a, _ := randDense(size, rho, randNormFloat32)
	d, _ := randDense(size, rho, randNormFloat32)
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		var n Dense
		n.Mul(a, d.T())
		wd = &n
	}
}
