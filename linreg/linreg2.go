// +build ignore

// Linreg2 reads lines with 2 floats x, y each until eof
// and outputs alpha, beta, and epsilon such that
// y = alpha + beta * x is the best fit and epsilon squared  is the average of the squared residuals
// use linreg instead, this file is left here for cross validation.
// Note that the input format differs from linreg.
package main

import (
	"fmt"
	"io"
	"log"
	"math"
)

func main() {

	var n, x, y, sumx, sumy, sumxx, sumyy, sumxy float64

	for {
		k, err := fmt.Scanln(&x, &y)
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Println(err)
		}
		if k != 2 {
			continue
		}

		n += 1
		sumx += x
		sumxx += x * x
		sumy += y
		sumyy += y * y
		sumxy += x * y
	}

	if sumxx == sumx*sumx {
		log.Fatalf("not enough x variance in %.0f data points", n)
	}

	fmt.Println(sumy, sumxy)
	fmt.Println(n, sumx)
	fmt.Println(sumx, sumxx)

	sumx /= n
	sumxx /= n
	sumy /= n
	sumyy /= n
	sumxy /= n

	beta := (sumxy - sumx*sumy) / (sumxx - sumx*sumx)
	alpha := sumy - beta*sumx
	eps2 := sumyy - sumy*sumy + beta*(sumxy-sumx*sumy)

	fmt.Printf("%g %g %g\n", alpha, beta, math.Sqrt(eps2))
}
