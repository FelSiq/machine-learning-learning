import numpy
from apyori import apriori

if __name__ == "__main__":
	import sys
	if len(sys.argv) < 2:
		print("usage: " + sys.argv[0] + " <data_filepath>",
			"[-sep data_separator, default is \",\"",
			"[-encoding data_encoding, default is \"utf-8\"",
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
