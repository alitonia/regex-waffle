import random


def readFile(filename):
	with open(filename, 'r') as f:
		match = [word for word in f.readline().split(", ")]
		# removing of new line from last word in M
		match[-1] = match[-1][:-1]
		unmatch = [word for word in f.readline().split(", ")]
	
	return match, unmatch


# word sets M and U
match, unmatch = readFile("../datasets/sample.txt")

from urllib.parse import urlparse, parse_qs

#  unmatch and match uri list
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


# frac_match


def unique(list1):
	# initialize a null list
	unique_list = []
	
	# traverse for all elements
	for x in list1:
		# check if exists in unique_list or not
		if x not in unique_list:
			unique_list.append(x)
	return unique_list


# # test uri
# match, unmatch = (
# 	dict_match['/tienda1/publico/anadir.jsp'].get('pos'),
# 	dict_match['/tienda1/publico/anadir.jsp'].get('neg')
# )

match, unmatch = (
	random.sample([x.query for x in frac_match if len(x.query) > 0], 100),
	random.sample([x.query for x in frac_unmatch if len(x.query) > 0], 100),
)
# print(match[:5], unmatch[:5])
# exit(0)
print("set M: ", match[:5])
print("set U: ", unmatch[:5])


# start declaration here
def starter(match, unmatch):
	num_m = len(match)
	num_u = len(unmatch)
	print(num_m)
	print(num_u)
	
	def charsInSet(wordSet):
		chars = []
		
		for word in wordSet:
			for c in word:
				if c not in chars:
					chars.append(c)
		
		chars.sort()
		
		return chars
	
	chars_in_M = charsInSet(match)
	print('charset', chars_in_M[:5])
	
	def makeRanges(chars_in_M):
		ranges = []
		done = False
		i = 0
		
		while i < len(chars_in_M) - 1:
			distance = 0
			for j in range(i + 1, len(chars_in_M)):
				if ord(chars_in_M[j]) - ord(chars_in_M[i]) == distance + 1:
					distance += 1
					# if range contains last character from chars_in_M,
					# we exit both loops (search is done)
					if j == (len(chars_in_M) - 1):
						ranges.append(chars_in_M[i] + '-' + chars_in_M[j])
						done = True
						break
				else:
					if chars_in_M[i] != chars_in_M[j - 1]:
						ranges.append(chars_in_M[i] + '-' + chars_in_M[j - 1])
					i = j
					break
			if done:
				break
		
		return ranges
	
	ranges = makeRanges(chars_in_M)
	print('ranges', ranges)
	
	def ngram(M, U):
		res = {}
		
		# length of n-grams is between 2 and 4
		for n in range(1, 10):
			# we go through all elements in M
			for i in range(0, len(M)):
				word_m = M[i]
				
				# n-grams from current element
				ngrams_m = zip(*[word_m[i:] for i in range(n)])
				gram_m = ["".join(gr) for gr in ngrams_m]
				
				# we need set of n-grams
				gram_m = set(gram_m)
				
				# we update score for n-gram by +1, if it can be found in M
				for g in gram_m:
					if g not in res:
						res[g] = 1
					elif g in res:
						res[g] += 1
			
			# we go through all elements in U
			for j in range(0, len(U)):
				word_u = U[j]
				
				ngrams_u = zip(*[word_u[j:] for j in range(n)])
				gram_u = ["".join(gr) for gr in ngrams_u]
				gram_u = set(gram_u)
				
				# we update score for n-gram by -1, if it can be found in U
				for g in gram_u:
					if g not in res:
						res[g] = -1
					elif g in res:
						res[g] -= 1
		
		return res
	
	ngrams = ngram(match, unmatch)
	ngrams = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
	
	# ngram_subset is the smallest subset of n-grams
	# for which total score equals at least |M|
	# (|M| = number of elements in M)
	ngram_subset = []
	if len(ngrams) == 1:
		ngram_subset = [ngrams[0]]
	else:
		score = 0
		
		for i in range(len(ngrams)):
			if ngrams[i][1] > 0:  # we update only if score is positive
				score += ngrams[i][1]
				ngram_subset.append(ngrams[i][0])
				
				if score >= num_m:
					break
	
	print('ngram_subset', ngram_subset)
	
	FUNCTION_SET = [".*", ".+", ".?", ".{.,.}+",  # possessive quantifiers
	                "(.)",  # group
	                "[.]",  # character class
	                "[^.]",  # negated character
	                "..",  # concatenator (binary node)
	                ".|.",  # disjunction
	                ]
	
	TERMINAL_SET = ["a-z", "A-Z", "0-9", "^", "$", "%",  # instance independent terminals
	                "\w", "\W", "\d", "\D", "\b", "\B", "\A", "\Z", "\s", "\S",
	                "^\x00-\x7F",
	                '__'  # emptiness
	                ]
	
	BASING = [
		# "\+",
		# '[a-zA-Z]+',
		# '[a-zA-Z%0-9]+',
		# '\d+',
		# '\w+'
		'\d+'
	]
	
	# chars_in_M set goes in Terminal set
	TERMINAL_SET.extend(chars_in_M)
	
	# ngram_subset goes in Terminal set
	TERMINAL_SET.extend(ngram_subset)
	
	# ranges go in Terminal set
	TERMINAL_SET.extend(ranges)
	
	print('TERMINAL_SET', TERMINAL_SET)
	
	import re
	import random
	
	def getRandom():
		pickSet = random.choice(['f', 't'])
		if pickSet == 't':
			value = random.choice(FUNCTION_SET)
			if value in [".{.,.}+"]:
				return value, 3
			elif value in [".|.", ".."]:
				return value, 2
			else:
				return value, 1
		else:
			value = random.choice(TERMINAL_SET)
			return value, 0
	
	class Node(object):
		def __init__(self, depth, root, value=None):
			self.depth = depth
			self.value = ""
			self.childrenNum = 0
			self.id = -1
			
			if root:
				self.value = "."
				self.childrenNum = 2
			else:
				self.value, self.childrenNum = getRandom()

			
			self.left = None
			self.right = None
			self.third = None
			
			if self.value == '__':
				self.childrenNum = 0
			elif self.childrenNum == 3:
				self.left = Node(depth + 1, False)
				self.right = Node(depth + 1, False)
				self.third = Node(depth + 1, False)
			elif self.childrenNum == 2:
				self.left = Node(depth + 1, False)
				self.right = Node(depth + 1, False)
			elif self.childrenNum == 1:
				self.left = Node(depth + 1, False)
	
	def unvisit(n):
		q = []
		q.append(n)
		
		while len(q) >= 1:
			top = q.pop(0)
			
			if top.id > -1:
				top.id = -1
				i = top.childrenNum
				if i == 1:
					q.append(top.left)
				
				elif i == 2:
					q.append(top.left)
					q.append(top.right)
				
				elif i == 3:
					q.append(top.left)
					q.append(top.right)
					q.append(top.third)
	
	def treeNumeration(n):
		unvisit(n)
		
		parentMap = {}
		q = []
		q.append(n)
		num = 0
		
		while len(q) >= 1:
			top = q.pop(0)
			
			if top.id == -1:
				top.id = num
				num += 1
				i = top.childrenNum
				if i == 0:
					parentMap[num - 1] = [-1]
				elif i == 1:
					q.append(top.left)
					parentMap[num - 1] = [top.left]
				elif i == 2:
					q.append(top.left)
					q.append(top.right)
					parentMap[num - 1] = [top.left, top.right]
				else:
					q.append(top.left)
					q.append(top.right)
					q.append(top.third)
					parentMap[num - 1] = [top.left, top.right, top.third]
		
		# in map is always first index of left,
		# then right (if exists) and finally
		# third (if exists) child
		return parentMap
	
	def treeToString(node):
		if node is None:
			return ''
		if node.value in TERMINAL_SET:
			if node.value == "%":
				return "."
			elif node.value == r'\%':
				return "%"
			elif node.value == '__':
				return ''
			return node.value
		
		rl = treeToString(node.left)
		rr = ''
		if node.childrenNum == 2:
			rr = treeToString(node.right)
		if node.childrenNum == 3:
			rr = treeToString(node.right)
			rt = treeToString(node.third)
		
		if node.value in FUNCTION_SET:
			if node.value == ".*":
				string = rl + "*"
				return string
			if node.value == ".+":
				string = rl + "+"
				return string
			if node.value == ".?":
				string = rl + "?"
				return string
			if node.value == "(.)":
				string = "(" + rl + ")"
				return string
			if node.value == "[.]":
				string = "[" + rl + "]"
				return string
			if node.value == "[^.]":
				string = "[^" + rl + "]"
				return string
			if node.value == "..":
				string = rl + rr
				return string
			if node.value == ".|.":
				string = rl + "|" + rr
				return string
			if node.value == ".{.,.}+":
				string = rl + "{" + rr + "," + rt + "}+"
				return string
		
		# root
		string = rl + rr
		
		return string
	
	class Individual:
		def __init__(self, setM, setU, value=None):
			
			# code is abstract tree which saves regex
			self.code = self.initialize(value)
			self.wi = 10
			# setM and setU are sets M and U from beginning
			self.setM = setM.copy()
			self.setU = setU.copy()
			
			# n_m - n_u - maximize
			self.fitnessFunction = self.calculateFitnessFunction()
			# length(r) - minimize
			self.fitnessRegex = self.calculateFitnessRegex()
			# final fitness = wi*(n_m - n_u) - length(r)
			# and it should be maximized
			self.fitness = self.finalFitness()
		
		def initialize(self, value=None):
			generated = False
			trial = 0
			while not generated:
				input_val = value if trial == 0 else None
				n = Node(0, True, input_val)
				treeString = treeToString(n)
				if value in BASING:
					print('it', input_val, value, treeString, trial)
				try:
					re.compile(treeString)
					# if compile doesn't throw exception,
					# we've got a valid regex and we accept
					# this individual
					generated = True
				except Exception:
					generated = False
					trial += 1
					if trial > 1000:
						raise Exception('Too much wrong regex')
			return n
		
		# check if current regex is valid
		def isFeasible(self):
			treeString = treeToString(self.code)
			try:
				re.compile(treeString)
				return True
			except Exception:
				return False
		
		def __lt__(self, other):
			# we want to maximize final fitness
			return self.fitness > other.fitness
		
		def __str__(self):
			treeString = treeToString(self.code)
			return treeString
		
		def calculateFitnessFunction(self):
			n_m = 0
			n_u = 0
			regex = treeToString(self.code)
			
			for wordM in self.setM:
				# matchM - list of strings that are matched
				matchM = re.findall(regex, wordM)
				
				foundM = False
				
				if matchM:
					for m in matchM:
						# in case of character | in regex,
						# m can have more elements
						for elem in m:
							# print(elem, wordM)
							if elem != "":
								if len(elem) == len(wordM) or elem in wordM:
									n_m += 1
									foundM = True
									break
						if foundM:
							break
			
			for wordU in self.setU:
				matchU = re.findall(regex, wordU)
				foundU = False
				
				if matchU:
					for m in matchU:
						for elem in m:
							if elem != "":
								if len(elem) == len(wordU) or elem in wordU:
									# tune this number ?
									n_u += 0.4
									foundU = True
									break
						if foundU:
							break
			
			return n_m - n_u
		
		def calculateFitnessRegex(self):
			regex = treeToString(self.code)
			return len(regex)
		
		def finalFitness(self):
			try:
				return self.wi * self.calculateFitnessFunction() - self.calculateFitnessRegex()
			except Exception:
				return -100000
	
	# (based on documentation)
	POPULATION_SIZE = 10  # may adjust with size
	GENERATIONS_NUM = 1000
	POPULATION_NUM = 8
	TOURNAMENT_SIZE = 7
	MUTATION_PROB = 0.6
	ELITIZM_SIZE = int(0.2 * POPULATION_SIZE)
	
	import copy
	
	def selection(population):
		betsFitness = float('-inf')
		bestIndex = -1
		
		for i in range(TOURNAMENT_SIZE):
			index = random.randrange(len(population))
			if population[index].fitness > betsFitness:
				betsFitness = population[index].fitness
				bestIndex = index
		
		return bestIndex
	
	def replace(root, position, child, address):
		red = []
		red.append(root)
		found = False
		
		while not found:
			node = red.pop(0)
			if node.id == position:
				found = True
				if child == 0:
					node.left = address
				elif child == 1:
					node.right = address
				else:
					node.third = address
			else:
				children = node.childrenNum
				if children == 1:
					red.append(node.left)
				elif children == 2:
					red.append(node.left)
					red.append(node.right)
				elif children == 3:
					red.append(node.left)
					red.append(node.right)
					red.append(node.third)
	
	def crossover(parent1, parent2, child1, child2):
		map1 = treeNumeration(parent1.code)
		map2 = treeNumeration(parent2.code)
		
		parent1Size = len(map1)
		parent2Size = len(map2)
		
		breakpoint = -1
		if parent1Size <= parent2Size:
			breakpoint = random.randrange(parent1Size)
		else:
			breakpoint = random.randrange(parent2Size)
		
		find = breakpoint
		
		if find == 0:
			# root is chosen
			child1.code = copy.deepcopy(parent2.code)
			child2.code = copy.deepcopy(parent1.code)
		else:
			child1.code = copy.deepcopy(parent1.code)
			child2.code = copy.deepcopy(parent2.code)
			
			unvisit(child1.code)
			unvisit(child2.code)
			
			# we know that nodes will have same numeration
			mapChild1 = treeNumeration(child1.code)
			mapChild2 = treeNumeration(child2.code)
			
			map1Keys = mapChild1.keys()
			map2Keys = mapChild2.keys()
			
			replaceAtPositionParent1 = -1
			childAdress1 = -1
			side1 = -1
			replaceAtPositionParent2 = -1
			childAdress2 = -1
			side2 = -1
			
			for i in map1Keys:
				children = mapChild1[i]
				index = 0
				for child in children:
					if child == -1:
						continue
					if find == child.id:
						# parent id of subtree we will change
						replaceAtPositionParent1 = i
						# subtree which we'll change
						childAdress1 = child
						# number to identify which child it is
						# left - 0, right - 1, third - 2
						side1 = index
					else:
						index += 1
			
			# same process
			for i in map2Keys:
				children = mapChild2[i]
				index = 0
				for child in children:
					if child == -1:
						continue
					if find == child.id:
						replaceAtPositionParent2 = i
						childAdress2 = child
						side2 = index
					else:
						index += 1
			
			replace(child1.code, replaceAtPositionParent1, side1, childAdress2)
			replace(child2.code, replaceAtPositionParent2, side2, childAdress1)
			
			if not child1.isFeasible():
				child1.code = copy.deepcopy(parent1.code)
			if not child2.isFeasible():
				child2.code = copy.deepcopy(parent2.code)
	
	def mutation(individual):
		q = random.random()
		
		if MUTATION_PROB > q:
			# we save current tree (code of individual)
			oldCode = copy.deepcopy(individual.code)
			mapaSuseda = treeNumeration(individual.code)
			choiceRange = len(mapaSuseda)
			
			index = random.randrange(choiceRange)
			
			# we search tree for a node with wanted index
			previousValue = ""
			found = False
			
			q = []
			q.append(individual.code)
			
			while not found:
				n = q.pop(0)
				if n.id == index:
					# we found a node
					found = True
					previousValue = n.value
					if n.value in FUNCTION_SET:
						# it's some inner node
						newValue = random.choice(FUNCTION_SET)
						n.value = newValue
						children = n.childrenNum
						
						if n.value in [".*", ".+", ".?", "(.)", "[.]", "[^.]"] and children != 1:
							n.right = None
							if children == 3:
								n.third = None
							n.childrenNum = 1
						elif n.value in ["..", ".|."] and children != 2:
							if children == 1:
								n.right = Node(n.depth + 1, False)
							else:
								# it has three children
								n.third = None
							n.childrenNum = 2
						elif n.value == ".{.,.}+" and children != 3:
							if children == 1:
								n.right = Node(n.depth + 1, False)
								n.third = Node(n.depth + 1, False)
							else:
								# it has two children
								n.third = Node(n.depth, False)
							n.childrenNum = 3
						
						if not individual.isFeasible():
							n.value = previousValue
							individual.code = oldCode
					else:
						# it's leaf node and we choose new value from Terminal set
						newValue = random.choice(TERMINAL_SET)
						n.value = newValue
						if not individual.isFeasible():
							n.value = previousValue
				else:
					children = n.childrenNum
					if children == 0:
						continue
					elif children == 1:
						q.append(n.left)
					elif children == 2:
						q.append(n.left)
						q.append(n.right)
					elif children == 3:
						q.append(n.left)
						q.append(n.right)
						q.append(n.third)
	
	def genetic_programming(match, unmatch):
		base = [Individual(match, unmatch, term) for term in BASING]
		population = [Individual(match, unmatch) for _ in range(POPULATION_SIZE - len(base))]
		population.extend(base)
		newPopulation = [Individual(match, unmatch) for _ in range(POPULATION_SIZE)]
		
		print('httt', [treeToString(x.code) for x in base])
		
		solutions = []
		
		for i in range(GENERATIONS_NUM):
			population.sort()
			newPopulation[:ELITIZM_SIZE] = population[:ELITIZM_SIZE]
			
			# if we found individual that satisfies condition:
			# num_m - num_u = num_m
			# we save it in solutions
			# (it is candidate for best solution)
			# print([treeToString(x.code) for x in solutions])
			
			# if treeToString(population[0].code) in BASING:
			print(treeToString(population[0].code), population[0].fitnessFunction)
			print('heh', [treeToString(x.code) for x in population])
			if i > 30:
				exit(0)
			if population[0].fitnessFunction == num_m:
				solutions.append(population[0])
			
			for j in range(ELITIZM_SIZE, POPULATION_SIZE, 2):
				parent1Index = selection(population)
				parent2Index = selection(population)
				
				crossover(population[parent1Index], population[parent2Index], newPopulation[j], newPopulation[j + 1])
				
				mutation(newPopulation[j])
				mutation(newPopulation[j + 1])
				
				newPopulation[j].fitness = newPopulation[j].finalFitness()
				newPopulation[j + 1].fitness = newPopulation[j + 1].finalFitness()
			
			population = newPopulation
		
		# if we didn't save any individual in solution,
		# we take the best one from current population
		if len(solutions) == 0:
			population.sort()
			solutions.append(population[0])
		
		return solutions
	
	res = []
	last_top = None
	same_last_counting = 0
	EXCESSIVE = 4
	
	for i in range(POPULATION_NUM):
		print(str(i + 1) + ". population out of " + str(POPULATION_NUM))
		res.append(genetic_programming(match, unmatch))
		res.sort()
		current_top = res[0][0]
		print("Current best solution: ", res[0][0], res[0][0].fitness)
		if current_top == last_top and res[0][0].fitness != -1:
			same_last_counting += 1
			if same_last_counting >= EXCESSIVE:
				break
		else:
			last_top = current_top
			same_last_counting = 0
		print(last_top, same_last_counting)
	
	print("Current best solution: ", last_top)
	return last_top


print('----')

match_parsed = [parse_qs(x) for x in match]
unmatch_parsed = [parse_qs(x) for x in unmatch]

_match_keys = [[t for t in x.keys()] for x in match_parsed]
match_keys = []
for keys in _match_keys:
	for key in keys:
		if key not in match_keys:
			match_keys.append(key)

match_entries = dict()
unmatch_entries = dict()

for key in match_keys:
	for d in match_parsed:
		if d.get(key) is not None and match_entries.get(key) is None and ''.join(d[key]).isprintable():
			match_entries[key] = [d[key]]
		else:
			if d.get(key) is not None and d[key] not in match_entries[key] and ''.join(d[key]).isprintable():
				match_entries[key].append(d[key])
	
	for d in unmatch_parsed:
		if (
				unmatch_entries.get(key) is None
				and d.get(key) is not None
				and d[key] not in (match_entries.get(key) or [])
				and ''.join(d[key]).isprintable()
		):
			unmatch_entries[key] = [d[key]]
		else:
			if (
					d.get(key) is not None
					and d[key] not in (unmatch_entries.get(key) or [])
					and d[key] not in (match_entries.get(key) or [])
					and ''.join(d[key]).isprintable()
			):
				unmatch_entries[key].append(d[key])

print(match_entries.__len__())
print(unmatch_entries.__len__())

for key in match_keys:
	focused_match = [' '.join(x) for x in match_entries.get(key)] or []
	# focused_unmatch = [' '.join(x) for x in unmatch_entries.get(key)] or []
	focused_unmatch = []
	
	print(key, focused_match[:10], focused_unmatch[:10])
	if focused_match.__len__() == 1:
		result = focused_match[0]
	else:
		result = starter(focused_match, focused_unmatch)
	print('--->', key, result)
