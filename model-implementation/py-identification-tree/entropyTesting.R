# R code for disorder testing with different bases for a binary dataset 
disorder <- function(p, base=2) {
	if (p == 0 | p == 1.0)
		return 0.0
	return (-p *log(p, base) -(1.0 - p) * log(1.0 - p, base))
}

e = (1 + 1/999999)^999999


xvals = seq(0, 1.0, 1/10000)
plot(xvals, disorder(xvals, 2), col='red', type='l')
lines(xvals, disorder(xvals, e), col='green', type='l')
lines(xvals, disorder(xvals, 10), col='blue', type='l')
