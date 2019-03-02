// Equal.... functions have been adapted from gonum/floats
package mat32

import "github.com/chewxy/math32"

// EqualWithinRel returns true if the difference between a and b
// is not greater than tol times the greater value.
func EqualWithinRel(a, b, tol float32) bool {
	if a == b {
		return true
	}
	return math32.Abs(a-b) <= tol*math32.Max(math32.Abs(a), math32.Abs(b))
}

// EqualWithinAbsOrRel returns true if a and b are equal to within
// the absolute tolerance.
func EqualWithinAbsOrRel(a, b, absTol, relTol float32) bool {
	if a == b || math32.Abs(a-b) <= absTol {
		return true
	}
	return EqualWithinRel(a, b, relTol)
}
