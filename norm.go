package mat32

import (
	"fmt"

	"github.com/chewxy/math32"
)

func Norm(m Matrix, norm float32) float32 {
	rows, cols := m.Dims()
	if rawmatrixer, ok := m.(RawMatrixer); ok {
		rm := rawmatrixer.RawMatrix()
		if rm.Data == nil {
			panic("matrix: dimension mismatch")
		}
	}
	if tri, ok := m.(*TriDense); ok {
		if tri.mat.Data == nil {
			panic("matrix: dimension mismatch")
		}
	}
	if vd, ok := m.(*VecDense); ok {
		if vd.mat.Data == nil {
			panic("matrix: dimension mismatch")
		}
	}
	var s float32
	switch norm {
	case 1:
		for c := 0; c < cols; c++ {
			sCol := float32(0)
			for r := 0; r < rows; r++ {
				sCol += math32.Abs(m.At(r, c))
			}
			if s < sCol {
				s = sCol
			}
		}
	case 2:
		for r := 0; r < rows; r++ {

			for c := 0; c < cols; c++ {
				v := m.At(r, c)
				s += v * v
			}
		}
		s = math32.Sqrt(s)
	case math32.Inf(1):
		for r := 0; r < rows; r++ {
			sRow := float32(0)
			for c := 0; c < cols; c++ {
				sRow += math32.Abs(m.At(r, c))
			}
			if s < sRow {
				s = sRow
			}
		}
		return s
	default:
		panic(fmt.Errorf("unimplemented norm %v", norm))
	}
	return s
}
