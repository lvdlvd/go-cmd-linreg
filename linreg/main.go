// Linreg reads lines with k+1 floats y, x_0... x_{k-1} each, until eof
// and outputs beta_0... beta_{k-1} such that
// < [y - beta ⋅ x]^2 > is minimized over the read dataset.
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
	fGpl = flag.Bool("g", false, "print result as gnuplottable formula")
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

	var (
		n, k int
		yxj  []float64
		xxT  *mat64.SymDense
	)

	scanner := bufio.NewScanner(os.Stdin)
	for line := 1; scanner.Scan(); line++ {
		flds, err := parseFloats(strings.Fields(scanner.Text()))
		if err != nil {
			log.Printf("line %d: %v", line, err)
			continue
		}
		if k == 0 {
			if len(flds) < 2 {
				log.Printf("line %d: expecting at least 2 fields, got %d", line, len(flds))
				continue
			}
			k = len(flds) - 1
			yxj = make([]float64, k)
			xxT = mat64.NewSymDense(k, nil)

		} else if k+1 != len(flds) {
			log.Printf("line %d: expecting %d fields, got %d", line, k+1, len(flds))
			continue
		}

		n++

		xxT.SymRankOne(xxT, 1, mat64.NewVector(k, flds[1:]))
		for i, v := range flds[1:] {
			yxj[i] += flds[0] * v
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	if n == 0 {
		log.Fatal("No input.")
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
		p := 0
		for i := 0; i < k; i++ {
			fmt.Printf("%+f ", beta.At(i, 0))
			switch p {
			case 0:
			case 1:
				fmt.Printf("*x ")
			default:
				fmt.Printf("*x**%d ", p)
			}
			p += 1
		}
	} else {
		fmt.Println(mat64.Formatted(&beta))
	}
}
