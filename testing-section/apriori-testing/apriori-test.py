from apyori import apriori
import matplotlib.pyplot as plt

if __name__ == "__main__":
	import sys
	if len(sys.argv) < 2:
		print("usage: " + sys.argv[0] + " <data_filepath>",
			"[-sep data_separator, default is \",\"]",
			"[-encoding data_encoding, default is \"utf-8\"]",
			"[-supp minimal_support_value, default is 0.1]",
			"[-lift minimal_lift_value, default is 0]",
			"[-conf minimal_confidence_value, default is 0.1]", sep="\n\t")
		exit(1)

	try:
		sep = sys.argv[1 + sys.argv.index("-sep")]
	except:
		sep = ","

	try:
		support = float(sys.argv[1 + sys.argv.index("-supp")])
	except:
		support = 0.1

	try:
		confidence = float(sys.argv[1 + sys.argv.index("-conf")])
	except:
		confidence = 0.1

	try:
		lift = float(sys.argv[1 + sys.argv.index("-lift")])
	except:
		lift = 0.0

	try:
		encoding = sys.argv[1 + sys.argv.index("-encoding")]
	except:
		encoding = "utf-8"

	data = []
	with open(sys.argv[1], encoding=encoding) as f:
		for line in f:
			data.append(line.strip().split(sep=sep))

	results = list(apriori(data, 
		min_support=support, 
		min_confidence=confidence, 
		min_lift=lift))

	results.sort(key = lambda k: k[2][0][3], reverse=True)

	for res in results:
		print("Items= [", ", ".join(res[0]), 
			"]\n\tLift=", res[2][0][3], end="\n\n")

	print("total rules generated:", len(results))

	"""
	x=[]
	y=[]

	for val in range(10, 251, 5):
		x_val = val/1000
		y_val = len(list(apriori(data,
			min_support=x_val,
			min_confidence=x_val,
			min_lift=lift)))
		x.append(x_val)
		y.append(y_val)

	plt.subplot(2, 2, 1)
	plt.title("[0.010, 0.250]")
	plt.plot(x, y)
	plt.xlabel("Suporte/Confiança mínimos")
	plt.ylabel("Regras aceitas")
	plt.subplot(2, 2, 2)
	plt.title("[0.015, 0.250]")
	plt.plot(x[1:], y[1:])
	plt.xlabel("Suporte/Confiança mínimos")
	plt.ylabel("Regras aceitas")
	plt.subplot(2, 2, 3)
	plt.title("[0.020, 0.250]")
	plt.plot(x[2:], y[2:])
	plt.xlabel("Suporte/Confiança mínimos")
	plt.ylabel("Regras aceitas")
	plt.subplot(2, 2, 4)
	plt.title("[0.025, 0.250]")
	plt.xlabel("Suporte/Confiança mínimos")
	plt.ylabel("Regras aceitas")
	plt.plot(x[3:], y[3:])
	plt.subplots_adjust(hspace=0.5, wspace=0.25)
	plt.show()


	x=[]
	y=[]

	for val in range(16):
		x_val = val
		y_val = len(list(apriori(data,
			min_support=0.012,
			min_confidence=0.012,
			min_lift=val)))
		x.append(x_val)
		y.append(y_val)

	plt.plot(x, y)
	plt.title("Regras aceitas por Lift Mínimo")
	plt.xlabel("Lift mínimo")
	plt.ylabel("Regras aceitas")
	plt.show()
	"""
