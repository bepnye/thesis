import os, operator, re
import cPickle as pkl
import numpy as np
from nltk.tokenize import word_tokenize
from collections import Counter
from random import randint, shuffle
from scipy import sparse
from sklearn import linear_model
from string import replace

TAX = 0
GEN = 1
SMM = 2
LEN = 3
YEAR = 4

WORLD = 0
SPORTS = 1
BUSINESS = 2
ARTS = 3
US = 4

LONG = 0
SHORT = 1
FILES = 2

KEY = 0
VAL = 1

INC = 1.0
DEC = -1.0

UNK = "<UNK>"

tags = ["Top/News/World", "Top/News/Sports", "Top/News/Business", "Top/Features/Arts", "Top/News/U.S."]
tag_shorts = { "Top/News/World" : "world",
               "Top/News/Sports" : "sports",
               "Top/News/Business" : "business",
               "Top/Features/Arts" : "arts",
               "Top/News/U.S." : "US" }

def dict_inc(d, e):
  d[e] = d.get(e, 0) + 1

def get_sub_directories(top):
	subs = [os.path.join(top, sub) for sub in os.listdir(top)]
	return [s for s in subs if os.path.isdir(s)]

def get_all_files(top):
	subdirs = get_sub_directories(top)
	files = [os.path.join(top, sub) for sub in os.listdir(top) if (os.path.join(top, sub) not in subdirs) and sub[-3:] == "xml"]
	return sum([get_all_files(s) for s in subdirs], files)

def pkl_load(f):
  print "loading "+f+"..."
  d = pkl.load(open("pkls/"+f+".pkl", "r"))
  print "done"
  return d

def pkl_dump(d, f):
  print "dumping "+f+"..."
  pkl.dump(d, open("pkls/"+f+".pkl", "w", -1))
  print "done"

def get_counters(year):
  tax = {}
  gen = {}
  for article in year.keys():
    for tag in year[article][TAX]:
      tax[tag] = tax.get(tag, 0) + 1
    for tag in year[article][GEN]:
      gen[tag] = gen.get(tag, 0) + 1
  return tax, gen

def sorted_counter(c, idx = VAL, coeff = INC):
  return sorted([(k,c[k]) for k in c], key=lambda x: coeff*x[idx])

def get_top_tags(counter, n):
  return sorted_counter(counter)[:n]

def get_tag_values(lines, tag):
  vals = ()
  for l in lines:
    if tag in l:
      vals = vals + (l[l.index(">")+1:l.index("</")],)
  return vals
