/*
Interpol reads a dataset file named as the commandline argument consisting
of whitespace separated lines of the form
	x y1 ....
where x and y1... are numbers. empty lines and lines starting with a hash are ignored.

It then reads a similar file on stdin, only using the first column, x', to interpolate
and print y1'... with an n't degree polynomial.


test:
	seq 0 10 | awk '{print $1, $1*$1, $1*$1*$1}' > dataset.dat
	seq 0 0.5 10 | awk '{print $1, $1*$1, $1*$1*$1}' > test.dat
	interpol -n N dataset.dat < test.dat | diff test.dat -
should show no diffs for n>3
*/
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
)

var (
	degree = flag.Int("n", 1, "degree of interpolation, 0 = lookup nearest, 1 = linear (2 points), 2 = quadratic (3 points)...")
	clip   = flag.Bool("clip", false, "do not extrapolate outside of datasets bounds.")
)

func usage() {
	log.Println("usage: interpol [-n 1] dataset.dat < xfile.dat > out.dat")
	flag.PrintDefaults()
	os.Exit(1)
}

func main() {
	flag.Usage = usage
	flag.Parse()
	if *degree < 0 {
		log.Fatalf("degree -n %d: must be nonnegative", *degree)
	}

	if len(flag.Args()) != 1 {
		flag.Usage()
	}

	dataset, err := readDataset(flag.Arg(0))
	if err != nil {
		log.Fatalln(err)
	}
	if len(dataset) == 0 {
		log.Fatalf("%s contains no data", flag.Arg(0))
	}
	log.Printf("dataset %d records of %d columns.", len(dataset), len(dataset[0]))

	if len(dataset) < *degree {
		log.Fatalf("not enough data points for degree %d interpolation", *degree)
	}

	if !sort.IsSorted(byFirstCol(dataset)) {
		log.Println("sorting dataset...")
		sort.Sort(byFirstCol(dataset))
	}

	for i, _ := range dataset[1:] {
		if dataset[i+1][0] == dataset[i][0] {
			log.Println(dataset[i])
			log.Println(dataset[i+1])
			log.Fatalln("duplicate x values")
		}
	}

	s := bufio.NewScanner(os.Stdin)
	for line := 1; s.Scan(); line++ {
		if s.Text() == "" || strings.HasPrefix(s.Text(), "#") {
			continue
		}
		x, err := strconv.ParseFloat(strings.Fields(s.Text())[0], 64)
		if err != nil {
			log.Printf("line %d: %v", line, err)
			continue
		}

		if *clip && (x < dataset[0][0] || x > dataset[len(dataset)-1][0]) {
			continue
		}

		fmt.Print(x)
		for _, y := range interpolate(dataset, x, *degree) {
			fmt.Print(" ")
			fmt.Print(y)
		}
		fmt.Println()

	}
	if err := s.Err(); err != nil {
		log.Fatalln(err)
	}
}

func readDataset(fname string) ([][]float64, error) {
	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	var r [][]float64
	for line := 1; s.Scan(); line++ {
		if s.Text() == "" || strings.HasPrefix(s.Text(), "#") {
			continue
		}
		flds := strings.Fields(s.Text())
		if len(flds) < 2 {
			log.Println("short line %d", line)
			continue
		}
		if len(r) > 0 && len(r[0]) != len(flds) {
			log.Println("ragged input line %d, %d fields instead of %d", line, len(flds), len(r[0]))
			continue
		}

		rec := make([]float64, len(flds))
		var err error
		for i, _ := range flds {
			rec[i], err = strconv.ParseFloat(flds[i], 64)
			if err != nil {
				log.Printf("line %d column %d: %v", line, i+1, err)
			}
		}
		r = append(r, rec)
	}
	return r, s.Err()
}

type byFirstCol [][]float64

func (r byFirstCol) Len() int           { return len(r) }
func (r byFirstCol) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r byFirstCol) Less(i, j int) bool { return r[i][0] < r[j][0] }

func interpolate(d [][]float64, x float64, deg int) []float64 {

	// smallest i in [0..len] such that x < d[i]. if i == len, d[len-1] < x
	i := sort.Search(len(d), func(i int) bool { return x < d[i][0] })

	if deg == 0 {
		if i > 0 && (i == len(d) || (x-d[i-1][0] < d[i][0]-x)) {
			i--
		}
		return d[i][1:]
	}

	i_min := i - 1 - deg // index of the first candidate
	i_max := i + deg     // index of the last candidate

	if i_min < 0 {
		i_min = 0
		i_max = deg + 1
	}
	if i_max >= len(d) {
		i_max = len(d) - 1
		i_min = i_max - 1 - deg
	}

	// degree == 1: linear interpolation between 2 points
	for i_max-i_min > deg {
		dmin, dmax := math.Abs(x-d[i_min][0]), math.Abs(x-d[i_max][0])
		if dmin < dmax {
			i_max--
		} else {
			i_min++
		}

	}
	// now we have the deg+1 points closests to x

	var lj []float64
	for j := i_min; j <= i_max; j++ {
		lj = append(lj, 1.0)
		for i := i_min; i <= i_max; i++ {
			if i == j {
				continue
			}
			lj[j-i_min] *= (x - d[i][0]) / (d[j][0] - d[i][0])
		}
	}

	yy := make([]float64, len(d[0])-1)

	for k, _ := range yy {
		for j := i_min; j <= i_max; j++ {
			yy[k] += d[j][k+1] * lj[j-i_min]
		}
	}

	return yy
}
