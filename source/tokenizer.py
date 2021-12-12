from urllib.parse import urlparse, parse_qs
import re


def readFile(filename):
	with open(filename, 'r') as f:
		match = [word for word in f.readline().split(", ")]
		# removing of new line from last word in M
		match[-1] = match[-1][:-1]
		unmatch = [word for word in f.readline().split(", ")]
	
	return match, unmatch


match, unmatch = readFile("../datasets/sample.txt")

frac_match = [urlparse(x) for x in match]
dict_match = dict()

for link in frac_match:
	if dict_match.get(link.path) is None:
		dict_match[link.path] = dict()
		dict_match[link.path]['pos'] = [link.query]
	else:
		try:
			if dict_match[link.path]['pos'].index(link.query):
				pass
		except ValueError:
			dict_match[link.path]['pos'].append(link.query)

#
frac_unmatch = [urlparse(x) for x in unmatch]

for link in frac_unmatch:
	if dict_match.get(link.path) is None:
		dict_match[link.path] = dict()
		dict_match[link.path]['neg'] = [link.query]
	elif dict_match.get(link.path).get('neg') is None:
		dict_match[link.path]['neg'] = [link.query]
	else:
		try:
			if dict_match[link.path]['neg'].index(link.query):
				pass
		except ValueError:
			dict_match[link.path]['neg'].append(link.query)

# print(frac_match[:5])

patterns = [
	'[a-zA-Z0-9]+'
]

inclusion = []
exclusion = []

for entry in frac_match:
	for p in patterns:
		inclusion.extend(re.findall(p, entry.query))

for entry in frac_unmatch:
	for p in patterns:
		exclusion.extend(re.findall(p, entry.query))

print('here')
final = [x for x in inclusion if x not in exclusion]
print('there')

with open('test.txt', 'w+') as f:
	f.write(', '.join(final))
