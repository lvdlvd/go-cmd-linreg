// Polyreg reads lines with 2 floats y, x, until eof
// and outputs beta_0... beta_{k-1} such that
// < [y - beta_i ⋅ x^i]^2 > is minimized over the read dataset.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"

	"github.com/gonum/matrix/mat64"
)

var (
	fGpl  = flag.Bool("g", false, "print result as gnuplottable formula")
	fKmin = flag.Int("kmin", 0, "min order of polynomial")
	fK    = flag.Int("k", 2, "max order of polynomial")
	fEven = flag.Bool("e", false, "even powers only")
	fOdd  = flag.Bool("o", false, "odd powers only")
	fXY   = flag.Bool("xy", false, "input x, y pairs (default: y x pairs)")
)

func parseFloats(fld []string) ([]float64, error) {
	var r []float64
	for _, v := range fld {
		n, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return nil, err
		}
		r = append(r, n)
	}
	return r, nil
}

func main() {

	flag.Parse()

	if *fOdd && *fEven {
		log.Fatal("specify at most one of the flags -o or -e.")
	}

	k := *fK + 1
	if *fOdd || *fEven {
		k = (k + 1) / 2
	}

	xp := make([]float64, k)
	xp[0] = 0

	yxj := make([]float64, k)
	xxT := mat64.NewSymDense(k, nil)

	n := 0
	scanner := bufio.NewScanner(os.Stdin)
	for line := 1; scanner.Scan(); line++ {
		flds, err := parseFloats(strings.Fields(scanner.Text()))
		if err != nil {
			log.Printf("line %d: %v", line, err)
			continue
		}

		if len(flds) != 2 {
			log.Printf("line %d: expecting at 2 fields, got %d", line, len(flds))
			continue
		}

		if *fXY {
			flds[0], flds[1] = flds[1], flds[0]
		}

		n++

		x := flds[1]
		xx := 1.0

		switch {
		case *fOdd:
			xx = x
			fallthrough
		case *fEven:
			x *= x
		}

		for i := 0; i < k; i++ {
			xp[i] = xx
			xx *= x
		}

		xxT.SymRankOne(xxT, 1, mat64.NewVector(k, xp))
		for i, v := range xp {
			yxj[i] += flds[0] * v
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	if n == 0 {
		log.Fatal("No input.")
	}

	if *fXY {
		log.Printf("Read %d x-y pairs.", n)
	} else {
		log.Printf("Read %d y-x pairs.", n)
	}

	// fmt.Println("yxj = ", mat64.Formatted(mat64.NewVector(k, yxj), mat64.Prefix("       ")))
	// fmt.Println("xxT = ", mat64.Formatted(xxT, mat64.Prefix("       ")))

	var chol mat64.Cholesky
	if ok := chol.Factorize(xxT); !ok {
		log.Fatal("X matrix not semidefinite?")
	}

	var beta mat64.Dense
	beta.SolveCholesky(&chol, mat64.NewVector(k, yxj))

	// TODO(lvd): compute residual error
	if *fGpl {
		fmt.Println(mat64.Formatted(&beta))
	} else {

		p := 0
		pp := 1
		switch {
		case *fOdd:
			p = 1
			fallthrough
		case *fEven:
			pp = 2
		}
		for i := 0; i < k; i++ {
			fmt.Printf("%+f ", beta.At(i, 0))
			switch p {
			case 0:
			case 1:
				fmt.Printf("*x ")
			default:
				fmt.Printf("*x**%d ", p)
			}

			p += pp
		}
		fmt.Println()
	}
}
