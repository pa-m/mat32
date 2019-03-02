package mat32

import (
	"fmt"
	"math/rand"

	"github.com/chewxy/math32"
)

func floatsNorm(a []float32, norm float32) float32 {
	if norm != 2 {
		panic(fmt.Errorf("unimplemented norm %g", norm))
	}
	var s float32
	for _, v := range a {
		s += v * v
	}
	return math32.Sqrt(s)
}

// Equal returns true if the slices have equal lengths and
// all elements are numerically identical.
func floatsEqual(s1, s2 []float32) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, val := range s1 {
		if s2[i] != val {
			return false
		}
	}
	return true
}

// Same returns true if the input slices have the same length and the all elements
// have the same value with NaN treated as the same.
func floatsSame(s, t []float32) bool {
	if len(s) != len(t) {
		return false
	}
	for i, v := range s {
		w := t[i]
		if v != w && !(math32.IsNaN(v) && math32.IsNaN(w)) {
			return false
		}
	}
	return true
}

// EqualApprox returns true if the slices have equal lengths and
// all element pairs have an absolute tolerance less than tol or a
// relative tolerance less than tol.
func floatsEqualApprox(s1, s2 []float32, tol float32) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i, a := range s1 {
		if !EqualWithinAbsOrRel(a, s2[i], tol, tol) {
			return false
		}
	}
	return true
}

func randNormFloat32() float32 {
	return float32(rand.NormFloat64())
}
