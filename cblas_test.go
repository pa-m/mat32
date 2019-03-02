// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build cblas

package mat32

import (
	"gonum.org/v1/gonum/blas/blas32"
	"gonum.org/v1/netlib/blas"
)

func init() {
	blas32.Use(blas.Implementation{})
}
