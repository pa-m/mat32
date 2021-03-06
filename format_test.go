// Copyright ©2013 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mat32

import (
	"fmt"
	"testing"

	"github.com/chewxy/math32"
)

func TestFormat(t *testing.T) {
	type rp struct {
		format string
		output string
	}
	sqrt := func(_, _ int, v float32) float32 { return math32.Sqrt(v) }
	for i, test := range []struct {
		m   fmt.Formatter
		rep []rp
	}{
		// Dense matrix representation
		{
			Formatted(NewDense(3, 3, []float32{0, 0, 0, 0, 0, 0, 0, 0, 0})),
			[]rp{
				{"%v", "⎡0  0  0⎤\n⎢0  0  0⎥\n⎣0  0  0⎦"},
				{"% f", "⎡.  .  .⎤\n⎢.  .  .⎥\n⎣.  .  .⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:3, Cols:3, Stride:3, Data:[]float32{0, 0, 0, 0, 0, 0, 0, 0, 0}}, capRows:3, capCols:3}"},
				{"%s", "%!s(*mat32.Dense=Dims(3, 3))"},
			},
		},
		{
			Formatted(NewDense(3, 3, []float32{1, 1, 1, 1, 1, 1, 1, 1, 1})),
			[]rp{
				{"%v", "⎡1  1  1⎤\n⎢1  1  1⎥\n⎣1  1  1⎦"},
				{"% f", "⎡1  1  1⎤\n⎢1  1  1⎥\n⎣1  1  1⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:3, Cols:3, Stride:3, Data:[]float32{1, 1, 1, 1, 1, 1, 1, 1, 1}}, capRows:3, capCols:3}"},
			},
		},
		{
			Formatted(NewDense(3, 3, []float32{1, 1, 1, 1, 1, 1, 1, 1, 1}), Prefix("\t")),
			[]rp{
				{"%v", "⎡1  1  1⎤\n\t⎢1  1  1⎥\n\t⎣1  1  1⎦"},
				{"% f", "⎡1  1  1⎤\n\t⎢1  1  1⎥\n\t⎣1  1  1⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:3, Cols:3, Stride:3, Data:[]float32{1, 1, 1, 1, 1, 1, 1, 1, 1}}, capRows:3, capCols:3}"},
			},
		},
		{
			Formatted(NewDense(3, 3, []float32{1, 0, 0, 0, 1, 0, 0, 0, 1})),
			[]rp{
				{"%v", "⎡1  0  0⎤\n⎢0  1  0⎥\n⎣0  0  1⎦"},
				{"% f", "⎡1  .  .⎤\n⎢.  1  .⎥\n⎣.  .  1⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:3, Cols:3, Stride:3, Data:[]float32{1, 0, 0, 0, 1, 0, 0, 0, 1}}, capRows:3, capCols:3}"},
			},
		},
		{
			Formatted(NewDense(2, 3, []float32{1, 2, 3, 4, 5, 6})),
			[]rp{
				{"%v", "⎡1  2  3⎤\n⎣4  5  6⎦"},
				{"% f", "⎡1  2  3⎤\n⎣4  5  6⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:2, Cols:3, Stride:3, Data:[]float32{1, 2, 3, 4, 5, 6}}, capRows:2, capCols:3}"},
			},
		},
		{
			Formatted(NewDense(3, 2, []float32{1, 2, 3, 4, 5, 6})),
			[]rp{
				{"%v", "⎡1  2⎤\n⎢3  4⎥\n⎣5  6⎦"},
				{"% f", "⎡1  2⎤\n⎢3  4⎥\n⎣5  6⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:3, Cols:2, Stride:2, Data:[]float32{1, 2, 3, 4, 5, 6}}, capRows:3, capCols:2}"},
			},
		},
		{
			func() fmt.Formatter {
				m := NewDense(2, 3, []float32{0, 1, 2, 3, 4, 5})
				m.Apply(sqrt, m)
				return Formatted(m)
			}(),
			[]rp{
				{"%v", "⎡        0          1  1.4142135⎤\n⎣1.7320508          2   2.236068⎦"},
				{"%.2f", "⎡0.00  1.00  1.41⎤\n⎣1.73  2.00  2.24⎦"},
				{"% f", "⎡        .          1  1.4142135⎤\n⎣1.7320508          2   2.236068⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:2, Cols:3, Stride:3, Data:[]float32{0, 1, 1.4142135, 1.7320508, 2, 2.236068}}, capRows:2, capCols:3}"},
			},
		},
		{
			func() fmt.Formatter {
				m := NewDense(3, 2, []float32{0, 1, 2, 3, 4, 5})
				m.Apply(sqrt, m)
				return Formatted(m)
			}(),
			[]rp{
				{"%v", "⎡        0          1⎤\n⎢1.4142135  1.7320508⎥\n⎣        2   2.236068⎦"},
				{"%.2f", "⎡0.00  1.00⎤\n⎢1.41  1.73⎥\n⎣2.00  2.24⎦"},
				{"% f", "⎡        .          1⎤\n⎢1.4142135  1.7320508⎥\n⎣        2   2.236068⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:3, Cols:2, Stride:2, Data:[]float32{0, 1, 1.4142135, 1.7320508, 2, 2.236068}}, capRows:3, capCols:2}"},
			},
		},
		{
			func() fmt.Formatter {
				m := NewDense(2, 3, []float32{0, 1, 2, 3, 4, 5})
				m.Apply(sqrt, m)
				return Formatted(m, Squeeze())
			}(),
			[]rp{
				{"%v", "⎡        0  1  1.4142135⎤\n⎣1.7320508  2   2.236068⎦"},
				{"%.2f", "⎡0.00  1.00  1.41⎤\n⎣1.73  2.00  2.24⎦"},
				{"% f", "⎡        .  1  1.4142135⎤\n⎣1.7320508  2   2.236068⎦"},
				{"%#v", "&mat32.Dense{mat:blas32.General{Rows:2, Cols:3, Stride:3, Data:[]float32{0, 1, 1.4142135, 1.7320508, 2, 2.236068}}, capRows:2, capCols:3}"},
			},
		},
		{
			func() fmt.Formatter {
				m := NewDense(1, 10, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
				return Formatted(m, Excerpt(3))
			}(),
			[]rp{
				{"%v", "Dims(1, 10)\n[ 1   2   3  ...  ...   8   9  10]"},
			},
		},
		{
			func() fmt.Formatter {
				m := NewDense(10, 1, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
				return Formatted(m, Excerpt(3))
			}(),
			[]rp{
				{"%v", "Dims(10, 1)\n⎡ 1⎤\n⎢ 2⎥\n⎢ 3⎥\n .\n .\n .\n⎢ 8⎥\n⎢ 9⎥\n⎣10⎦"},
			},
		},
		{
			func() fmt.Formatter {
				m := NewDense(10, 10, nil)
				for i := 0; i < 10; i++ {
					m.Set(i, i, 1)
				}
				return Formatted(m, Excerpt(3))
			}(),
			[]rp{
				{"%v", "Dims(10, 10)\n⎡1  0  0  ...  ...  0  0  0⎤\n⎢0  1  0            0  0  0⎥\n⎢0  0  1            0  0  0⎥\n .\n .\n .\n⎢0  0  0            1  0  0⎥\n⎢0  0  0            0  1  0⎥\n⎣0  0  0  ...  ...  0  0  1⎦"},
			},
		},
	} {
		for j, rp := range test.rep {
			got := fmt.Sprintf(rp.format, test.m)
			if got != rp.output {
				t.Errorf("unexpected format result test %d part %d:\ngot:\n%s\nwant:\n%s", i, j, got, rp.output)
			}
		}
	}
}
