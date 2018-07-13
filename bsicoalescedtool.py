#!/usr/bin/env python2

import sys, os, struct
import codecs, re
import heapq

from collections import defaultdict, OrderedDict

DEBUG = 0
ENCODING = 1
ENDIAN = '<'

script_file_name = os.path.split(sys.argv[0])[1]
script_file_base = os.path.splitext(script_file_name)[0]

if len(sys.argv) < 2:
	print 'Coalesced tool for BioShock Infinite, (c) flatz'
	print 'Usage: {0} <command> [options]'.format(script_file_name)
	print 'Command    Options                                             Description'
	print '  unpack   <bin directory> <output directory>                  Unpack files'
	print '  pack     <listing file> <input directory> <output directory> Pack files'
	print '  list     <bin directory> <listing file>                      List files'
	sys.exit()
command = sys.argv[1]

def to_bits(s):
	result = ''
	for c in s:
		bits = bin(ord(c))[2:]
		bits = '00000000'[len(bits):] + bits
		result += bits
	return result

def from_bits(bits):
	bit_length = len(bits)
	assert bit_length % 8 == 0
	chars = ''
	for b in xrange(len(bits) / 8):
		chars += chr(int(bits[b * 8:(b + 1) * 8], 2))
	return ''.join(chars)

def has_leading_space(s):
	return re.match(r'^\s+', s) is not None

def has_quotes(s):
	return s[0] == '\"' and s[-1] == '\"' if len(s) >= 2 else False

def indentation(level):
	return '  ' * level

def strip_leading_zeroes(bits):
	return bits.lstrip('0')

def pad_leading_zeroes(bits):
	bit_length = len(bits)
	pad_length = ((bit_length + 7) // 8) * 8 - bit_length
	return '0' * pad_length + bits

def walk_directory(directory, callback):
	directory = os.path.abspath(directory)
	for entry_name in [entry_name for entry_name in os.listdir(directory) if not entry_name in ['.', '..']]:
		entry_path = os.path.join(directory, entry_name)
		callback(entry_path)
		if os.path.isdir(entry_path):
			walk_directory(entry_path, callback)

def read_u8(f):
	fmt = 'B'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def read_u16_le(f):
	fmt = '<H'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def read_s16_le(f):
	fmt = '<h'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def read_u32_le(f):
	fmt = '<I'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def read_s32_le(f):
	fmt = '<i'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def read_u16_be(f):
	fmt = '>H'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def read_s16_be(f):
	fmt = '>h'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def read_u32_be(f):
	fmt = '>I'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def read_s32_be(f):
	fmt = '>i'
	return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
def write_u8(f, data):
	f.write(struct.pack('B', data))
def write_u16_le(f, data):
	f.write(struct.pack('<H', data))
def write_s16_le(f, data):
	f.write(struct.pack('<h', data))
def write_u32_le(f, data):
	f.write(struct.pack('<I', data))
def write_s32_le(f, data):
	f.write(struct.pack('<i', data))
def write_u16_be(f, data):
	f.write(struct.pack('>H', data))
def write_s16_be(f, data):
	f.write(struct.pack('>h', data))
def write_u32_be(f, data):
	f.write(struct.pack('>I', data))
def write_s32_be(f, data):
	f.write(struct.pack('>i', data))

def read_string(f):
	count = read_s32_be(f) if ENDIAN == '>' else read_s32_le(f)
	is_unicode = count < 0
	if is_unicode:
		string = u''.join(unichr(read_u16_le(f)) for x in xrange(-count))
	else:
		string = ''.join(chr(read_u8(f)) for x in xrange(count))
	return string.rstrip('\0')

def write_string(f, string):
	count = len(string) + 1
	if isinstance(string, unicode):
		if ENDIAN == '>':
			write_s32_be(f, -count)
		else:
			write_s32_le(f, -count)
		for c in string:
			write_u16_le(f, ord(c))
		write_u16_le(f, 0)
	elif isinstance(string, str):
		if ENDIAN == '>':
			write_s32_be(f, count)
		else:
			write_s32_le(f, count)
		for c in string:
			write_u8(f, ord(c))
		write_u8(f, 0)

def read_data(f):
	count = read_s32_be(f) if ENDIAN == '>' else read_s32_le(f)
	assert count <= 0
	data = f.read(-count)
	return data

def write_data(f, data):
	count = len(data)
	if ENDIAN == '>':
		write_s32_be(f, -count)
	else:
		write_s32_le(f, -count)
	f.write(data)

def unescape(s, quotes=False):
	if quotes and has_quotes(s):
		s = s[1:-1]
	length = len(s)
	if length == 0:
		return s
	result = ''
	p = 0
	while p < length:
		if s[p] != '\\':
			result += s[p]
			p += 1
		else:
			if p + 1 < length and s[p + 1] in ['\"']: #['r', 'n', '\"']:
				if s[p + 1] == '\"':
					result += '\"'
					p += 2
				#elif s[p + 1] == 'r':
				#	result += '\r'
				#	p += 2
				#elif s[p + 1] == 'n':
				#	result += '\n'
				#	p += 2
				continue
			result += s[p]
			p += 1
	return result

def escape(s, quotes=False):
	result = ''
	for c in s:
		if c == '\r':
			result += '\\r'
		elif c == '\n':
			result += '\\n'
		elif c == '\"':
			result += '\\\"'
		else:
			result += c
	return '\"' + result + '\"' if quotes and has_leading_space(result) else result

class IniFile(object):
	SECTION_REGEXP = re.compile(r'\[\s*(.*?)\s*\]')
	KEY_VALUE_PAIR_REGEXP = re.compile(r'^\s*([+-.!]?)\s*(.*?)\s*=(.*?)$')
	KEY_VALUE_PAIR_WITHOUT_CMD_REGEXP = re.compile(r'^\s*(.*?)\s*=(.*?)$')

	class Section(object):
		def __init__(self, name):
			self.name = name
			self.pairs = []

		def add(self, key, value):
			self.pairs.append((key, value))

		def add_unique(self, key, value):
			for (k, v) in self.pairs:
				if k == key and v == value:
					return
			self.add(key, value)

		def remove(self, key):
			for (k, v) in self.pairs:
				if k == key:
					self.pairs.remove((k, v))

		def remove_pair(self, key, value):
			for (k, v) in self.pairs:
				if k == key and v == value:
					self.pairs.remove((k, v))

		def replace(self, key, value):
			found = False
			for index, (k, v) in enumerate(self.pairs):
				if k == key:
					self.pairs[index] = (k, v)
					found = True
			if not found:
				self.add(key, value)

	def __init__(self, enclose_in_quotes=False, default_cmd=None):
		assert default_cmd in ['+', '-', '!', '#', None]
		self.enclose_in_quotes = enclose_in_quotes
		self.default_cmd = default_cmd
		self.sections = None

	def load(self, f):
		self.sections = []
		current_section = None
		for line in f:
			line = line.rstrip('\r\n')
			if len(line) == 0 or line[0] == ';':
				continue
			m = self.SECTION_REGEXP.match(line)
			if m is not None:
				name = m.group(1)
				current_section = self.Section(name)
				self.sections.append(current_section)
				continue
			elif current_section is None:
				continue
			m = self.KEY_VALUE_PAIR_WITHOUT_CMD_REGEXP.match(line)
			if m is None:
				continue
			key, value = m.group(1), unescape(m.group(2), self.enclose_in_quotes)
			current_section.add_unique(key, value)
			"""
			cmd = m.group(1) if m.group(1) is not None else self.default_cmd
			key, value = m.group(2), unescape(m.group(3), self.enclose_in_quotes)
			if cmd == '+':
				current_section.add_unique(key, value)
			elif cmd == '-':
				current_section.remove_pair(key, value)
			elif cmd == '!':
				current_section.remove(key)
			else:
				current_section.replace(key, value)
			"""

	def save(self, f):
		for section in self.sections:
			f.write(u'[{0}]\r\n'.format(section.name))
			for (key, value) in section.pairs:
				f.write(u'{0}={1}\r\n'.format(key, escape(value, self.enclose_in_quotes)))
			f.write('\r\n')

class CoalescedFile(object):
	class CodeTable(object):
		class Node(object):
			def __init__(self, symbol=None, frequency=None):
				self.symbol = symbol
				self.frequency = frequency
				self.left = None
				self.right = None

			def set_children(self, left, right):
				self.left = left
				self.right = right

			def is_leaf(self):
				return self.symbol is not None

			def __lt__(self, other):
				return self.frequency < other.frequency

			def __repr__(self):
				if self.is_leaf():
					return 'Leaf(symbol={0}, frequency={1})'.format(self.symbol, self.frequency)
				else:
					return 'Branch(frequency={0})'.format(self.frequency)

		class Code(object):
			def __init__(self):
				self.symbol = None
				self.left = None
				self.right = None

			def load(self, f):
				if ENDIAN == '>':
					self.symbol = read_u16_be(f)
					self.left = read_s16_be(f)
					self.right = read_s16_be(f)
				else:
					self.symbol = read_u16_le(f)
					self.left = read_s16_le(f)
					self.right = read_s16_le(f)
				return True

			def save(self, f):
				if ENDIAN == '>':
					write_u16_be(f, self.symbol)
					write_s16_be(f, self.left)
					write_s16_be(f, self.right)
				else:
					write_u16_le(f, self.symbol)
					write_s16_le(f, self.left)
					write_s16_le(f, self.right)
				return True

			def dump(self, f, i, depth):
				print >> f, indentation(depth) + u'code: {0}: symbol:{1} left:{2} right:{3}'.format(i, self.symbol, self.left, self.right)

			def __repr__(self):
				return 'Code(symbol={0}, left={1}, right={2})'.format(self.symbol, self.left, self.right)

		def __init__(self):
			self.nodes = None
			self.tree = None
			self.codes = None

		def load(self, f):
			def build_tree(code):
				if code.left < 0 and code.right < 0:
					node = self.Node(code.symbol)
					self.nodes.append(node)
				else:
					node = self.Node()
					self.nodes.append(node)
					if code.left >= 0:
						node.left = build_tree(codes[code.left])
					if code.right >= 0:
						node.right = build_tree(codes[code.right])
				return node
			if ENDIAN == '>':
				num_codes = read_s32_be(f)
			else:
				num_codes = read_s32_le(f)
			assert num_codes >= 0
			codes = []
			for i in xrange(num_codes):
				code = self.Code()
				code.load(f)
				codes.append(code)
			self.nodes = []
			self.tree = build_tree(codes[0])
			self.codes = {}
			self.assign_code(self.tree, '')
			return True

		def save(self, f):
			def build_codes(node):
				if node.is_leaf():
					code = self.Code()
					code.symbol = node.symbol
					code.left = -1
					code.right = -1
					codes.append(code)
				else:
					code = self.Code()
					code.symbol = 0xFFFF
					code.left = self.nodes.index(node.left) if node.left is not None else -1
					code.right = self.nodes.index(node.right) if node.right is not None else -1
					codes.append(code)
					build_codes(node.left)
					build_codes(node.right)
			codes = []
			if self.tree is not None:
				node_index = 0
				build_codes(self.tree)
			num_codes = len(codes)
			if ENDIAN == '>':
				write_s32_be(f, num_codes)
			else:
				write_s32_le(f, num_codes)
			for i in xrange(num_codes):
				code = codes[i]
				code.save(f)
			return True

		def assign_code(self, node, prefix):
			if node.is_leaf():
				assert len(prefix) > 0
				self.codes[node.symbol] = prefix
			else:
				self.assign_code(node.left, prefix + '0')
				self.assign_code(node.right, prefix + '1')

		def compute(self, symbols):
			def build_nodes(node):
				if node.is_leaf():
					self.nodes.append(node)
				else:
					self.nodes.append(node)
					if node.left is not None:
						build_nodes(node.left)
					if node.right is not None:
						build_nodes(node.right)
			frequencies = defaultdict(int)
			for c in symbols:
				frequencies[ord(c)] += 1
			self.tree = [self.Node(symbol, frequency) for symbol, frequency in frequencies.items()]
			heapq.heapify(self.tree)
			while len(self.tree) > 1:
				right, left = heapq.heappop(self.tree), heapq.heappop(self.tree)
				branch = self.Node(None, left.frequency + right.frequency)
				branch.set_children(left, right)
				heapq.heappush(self.tree, branch)
			self.tree = heapq.heappop(self.tree)
			self.nodes = []
			build_nodes(self.tree)
			self.codes = {}
			self.assign_code(self.tree, '')

		def encode(self, string):
			bits = ''.join(self.codes[ord(c)] for c in string)
			return bits

		def decode(self, bits):
			data = ''
			node = self.tree
			for bit in bits:
				node = node.left if bit == '0' else node.right
				if node.is_leaf():
					data += unichr(node.symbol)
					node = self.tree
			return data

		def dump(self, f, depth):
			print >> f, indentation(depth) + 'codes:'
			for symbol in sorted(self.codes, key=lambda x: len(self.codes[x])):
				print >> f, indentation(depth + 1) + 'symbol:{0} code:{1}'.format(symbol, self.codes[symbol])

	class File(object):
		class Section(object):
			class Pair(object):
				def __init__(self):
					self.key = None
					self.value = None
					self.length = None
					self.bits = None

				def load(self, f):
					self.key = read_string(f)
					data = read_data(f)
					if ENDIAN == '>':
						self.length, data = (ord(data[0]) << 8) | ord(data[1]), data[2:]
					else:
						self.length, data = (ord(data[1]) << 8) | ord(data[0]), data[2:]
					if self.length > 0:
						self.bits = strip_leading_zeroes(to_bits(data[::-1]))[::-1]
					else:
						self.bits = ''
					return True

				def save(self, f):
					write_string(f, self.key)
					if ENCODING:
						if ENDIAN == '>':
							data = chr((self.length >> 8) & 0xFF) + chr(self.length & 0xFF)
						else:
							data = chr(self.length & 0xFF) + chr((self.length >> 8) & 0xFF)
						if self.length > 0:
							data += from_bits(pad_leading_zeroes(self.bits[::-1]))[::-1]
						else:
							pass
						write_data(f, data)
					else:
						write_string(f, self.value.rstrip('\0'))
					return True

				def decode(self, code_table):
					self.value = code_table.decode(self.bits) if self.length > 0 else ''

				def encode(self, code_table):
					self.bits = code_table.encode(self.value) if self.length > 0 else ''

				def dump(self, f, depth):
					if self.value is None:
						print >> f, indentation(depth) + u'key:{0} bits:{1}'.format(self.key, self.bits)
					else:
						value = escape(self.value.rstrip('\0'))
						print >> f, indentation(depth) + u'key:{0} value:{1}'.format(self.key, value)

			def __init__(self):
				self.name = None
				self.num_pairs = None
				self.pairs = None

			def load(self, f):
				self.name = read_string(f)
				if ENDIAN == '>':
					self.num_pairs = read_s32_be(f)
				else:
					self.num_pairs = read_s32_le(f)
				assert self.num_pairs >= 0
				self.pairs = []
				for i in xrange(self.num_pairs):
					pair = self.Pair()
					pair.load(f)
					self.pairs.append(pair)
				return True

			def save(self, f):
				write_string(f, self.name)
				if ENDIAN == '>':
					write_s32_be(f, self.num_pairs)
				else:
					write_s32_le(f, self.num_pairs)
				for i in xrange(self.num_pairs):
					pair = self.pairs[i]
					pair.save(f)
				return True

			def dump(self, f, depth):
				print >> f, indentation(depth) + 'name:{0} num_pairs:{1}'.format(self.name, self.num_pairs)
				print >> f, indentation(depth) + 'pairs:'
				for i in xrange(self.num_pairs):
					pair = self.pairs[i]
					pair.dump(f, depth + 1)

		def __init__(self):
			self.path = None
			self.num_sections = None
			self.sections = None

		def load(self, f):
			self.path = read_string(f)
			if ENDIAN == '>':
				self.num_sections = read_s32_be(f)
			else:
				self.num_sections = read_s32_le(f)
			assert self.num_sections >= 0
			self.sections = []
			for i in xrange(self.num_sections):
				section = self.Section()
				section.load(f)
				self.sections.append(section)
			return True

		def save(self, f):
			write_string(f, self.path)
			if ENDIAN == '>':
				write_s32_be(f, self.num_sections)
			else:
				write_s32_le(f, self.num_sections)
			for i in xrange(self.num_sections):
				section = self.sections[i]
				section.save(f)
			return True

		def dump(self, f, depth):
			print >> f, indentation(depth) + 'path:{0} num_sections:{1}'.format(self.path, self.num_sections)
			print >> f, indentation(depth) + 'sections:'
			for i in xrange(self.num_sections):
				section = self.sections[i]
				section.dump(f, depth + 1)

	def __init__(self, is_global=False):
		self.is_global = is_global
		self.code_table = None
		self.num_files = None
		self.files = None

	def load(self, f):
		if self.is_global:
			self.code_table = self.CodeTable()
			self.code_table.load(f)
		if ENDIAN == '>':
			self.num_files = read_s32_be(f)
		else:
			self.num_files = read_s32_le(f)
		assert self.num_files >= 0
		self.files = []
		for i in xrange(self.num_files):
			file = self.File()
			file.load(f)
			self.files.append(file)
		return True

	def save(self, f):
		if ENCODING:
			if self.is_global:
				self.code_table.save(f)
		if ENDIAN == '>':
			write_s32_be(f, self.num_files)
		else:
			write_s32_le(f, self.num_files)
		for i in xrange(self.num_files):
			file = self.files[i]
			file.save(f)
		return True

	def iterate_over_chars(self):
		values = self.iterate_over_values()
		for value in values:
			for c in value:
				yield c

	def iterate_over_values(self):
		for i in xrange(self.num_files):
			file = self.files[i]
			for j in xrange(file.num_sections):
				section = file.sections[j]
				for k in xrange(section.num_pairs):
					pair = section.pairs[k]
					if pair.value is None or pair.length == 0:
						continue
					yield pair.value

	def decode(self, code_table):
		for i in xrange(self.num_files):
			file = self.files[i]
			for j in xrange(file.num_sections):
				section = file.sections[j]
				for k in xrange(section.num_pairs):
					pair = section.pairs[k]
					pair.decode(code_table)

	def encode(self, code_table):
		for i in xrange(self.num_files):
			file = self.files[i]
			for j in xrange(file.num_sections):
				section = file.sections[j]
				for k in xrange(section.num_pairs):
					pair = section.pairs[k]
					pair.encode(code_table)

	def dump(self, f, depth=0):
		if self.is_global:
			print >> f, indentation(depth) + 'code_table:'
			self.code_table.dump(f, depth + 1)
		print >> f, indentation(depth) + 'num_files:{0}'.format(self.num_files)
		print >> f, indentation(depth) + 'files:'
		for i in xrange(self.num_files):
			file = self.files[i]
			file.dump(f, depth + 1)

if command == 'unpack':
	if len(sys.argv) < 4:
		print 'error: insufficient options specified'
		sys.exit()

	bin_directory = sys.argv[2]
	if not os.path.isdir(bin_directory):
		print 'error: invalid bin directory specified'
		sys.exit()
	output_directory = sys.argv[3]
	if os.path.exists(output_directory) and not os.path.isdir(output_directory):
		print 'error: invalid output directory specified'
		sys.exit()
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)

	code_table = None
	localization_files = {}
	bin_files = os.listdir(bin_directory)
	for file_name in bin_files:
		file_path = os.path.join(bin_directory, file_name)
		file_base, file_extension = os.path.splitext(file_name)
		if file_extension.upper() != '.BIN' or not os.path.isfile(file_path):
			continue
		coalesced_file = CoalescedFile(file_base.upper().startswith('COALESCED_'))
		with open(file_path, 'rb') as f:
			coalesced_file.load(f)
		if coalesced_file.is_global:
			if code_table is not None:
				print 'error: multiple code tables found'
				sys.exit()
			code_table = coalesced_file.code_table
		localization_files[file_name] = coalesced_file
	if len(localization_files) == 0:
		print 'error: no localization files found'
		sys.exit()

	if code_table is None:
		print 'error: no code table found'
		sys.exit()

	log_file = codecs.open('log.txt', 'w', 'utf-8') if DEBUG else None
	for file_name, coalesced_file in localization_files.items():
		coalesced_file.decode(code_table)
		if log_file is not None:
			log_file.write(u'{0}:\n'.format(file_name))
			coalesced_file.dump(log_file)
			log_file.write(u'\n--------------------\n\n')
		for i in xrange(coalesced_file.num_files):
			file = coalesced_file.files[i]
			file_path = os.path.join(output_directory, file.path.replace('..\\', '').replace('\\', os.sep))
			file_directory = os.path.split(file_path)[0]
			if not os.path.exists(file_directory):
				os.makedirs(file_directory)
			ini_file = IniFile(default_cmd='+')
			ini_file.sections = []
			for j in xrange(file.num_sections):
				section = file.sections[j]
				ini_section = IniFile.Section(section.name)
				for k in xrange(section.num_pairs):
					pair = section.pairs[k]
					key = pair.key
					value = pair.value[:pair.length - 1] if pair.length > 0 else ''
					ini_section.add_unique(key, value)
				ini_file.sections.append(ini_section)
			with codecs.open(file_path, 'w', 'utf-16') as f:
				ini_file.save(f)
	if log_file is not None:
		log_file.close()

	# FIXME
	#with open('code_table.bin', 'wb') as f:
	#	code_table.save(f)
elif command == 'pack':
	if len(sys.argv) < 5:
		print 'error: insufficient options specified'
		sys.exit()

	listing_file_path = sys.argv[2]
	if not os.path.isfile(listing_file_path):
		print 'error: invalid listing file specified'
		sys.exit()
	input_directory = sys.argv[3]
	if not os.path.isdir(input_directory):
		print 'error: invalid input directory specified'
		sys.exit()
	output_directory = sys.argv[4]
	if os.path.exists(output_directory) and not os.path.isdir(output_directory):
		print 'error: invalid output directory specified'
		sys.exit()
	if not os.path.exists(output_directory):
		os.makedirs(output_directory)

	localization_files = {}
	with codecs.open(listing_file_path, 'r', 'utf-8') as f:
		lines = f.readlines()
		for line in lines:
			line = line.rstrip('\r\n')
			parts = line.split(':', 1)
			if len(parts) < 2:
				continue
			file_name, ini_files = parts[0], filter(bool, parts[1].split(':'))
			file_base, file_extension = os.path.splitext(file_name)
			if file_extension.upper() != '.BIN':
				continue
			coalesced_file = CoalescedFile(file_base.upper().startswith('COALESCED_'))
			coalesced_file.files = []
			for ini_file_path in ini_files:
				file_path = os.path.join(input_directory, ini_file_path.replace('..\\', '').replace('\\', os.sep))
				if not os.path.isfile(file_path):
					print 'error: file not found:', file_path
					sys.exit()
				ini_file = IniFile(default_cmd='+')
				with codecs.open(file_path, 'r', 'utf-16') as f:
					ini_file.load(f)
				file = CoalescedFile.File()
				file.path = ini_file_path
				file.sections = []
				for ini_section in ini_file.sections:
					section = CoalescedFile.File.Section()
					section.name = ini_section.name
					section.pairs = []
					for (key, value) in ini_section.pairs:
						pair = CoalescedFile.File.Section.Pair()
						pair.key = key
						pair.length = len(value)
						if pair.length > 0:
							pair.value = value + '\0'
							pair.length += 1
						else:
							pair.value = ''
						section.pairs.append(pair)
					section.num_pairs = len(section.pairs)
					file.sections.append(section)
				file.num_sections = len(file.sections)
				coalesced_file.files.append(file)
			coalesced_file.num_files = len(coalesced_file.files)
			localization_files[file_name] = coalesced_file

	symbols = ''
	for file_name, coalesced_file in localization_files.items():
		symbols += ''.join(list(coalesced_file.iterate_over_chars()))

	code_table = CoalescedFile.CodeTable()
	code_table.compute(symbols)
	# FIXME
	#with open('code_table.bin', 'rb') as f:
	#	code_table.load(f)

	for file_name, coalesced_file in localization_files.items():
		if coalesced_file.is_global:
			coalesced_file.code_table = code_table
		coalesced_file.encode(code_table)
		file_path = os.path.join(output_directory, file_name)
		with open(file_path, 'wb') as f:
			coalesced_file.save(f)
elif command == 'list':
	if len(sys.argv) < 4:
		print 'error: insufficient options specified'
		sys.exit()

	bin_directory = sys.argv[2]
	if not os.path.isdir(bin_directory):
		print 'error: invalid bin directory specified'
		sys.exit()
	listing_file_path = sys.argv[3]
	if os.path.exists(listing_file_path) and not os.path.isfile(listing_file_path):
		print 'error: invalid listing file specified'
		sys.exit()

	localization_files = {}
	bin_files = os.listdir(bin_directory)
	for file_name in bin_files:
		file_path = os.path.join(bin_directory, file_name)
		file_base, file_extension = os.path.splitext(file_name)
		if file_extension.upper() != '.BIN' or not os.path.isfile(file_path):
			continue
		coalesced_file = CoalescedFile(file_base.upper().startswith('COALESCED_'))
		with open(file_path, 'rb') as f:
			coalesced_file.load(f)
		localization_files[file_name] = ':'.join([x.path for x in coalesced_file.files])
	if len(localization_files) == 0:
		print 'error: no localization files found'
		sys.exit()

	with codecs.open(listing_file_path, 'w', 'utf-8') as f:
		for file_name, files in sorted(localization_files.items()):
			f.write(u'{0}:{1}\n'.format(file_name, files))
else:
	print 'error: unknown command'
	sys.exit()
