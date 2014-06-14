from utils import *
from matrix import *
from files_to_structs import *


def get_smm_freq(year, tag, tag_idx):
  yes = 0.0
  total = 0.0
  for article in year.keys():
    if tag in year[article][tag_idx]:
      total += 1.0
      if year[article][SMM]:
        yes += 1.0
  return yes/total, total

def make_all_years():
  all_years = {}
  for year in [2005, 2006, 2007]:
    print "Working on year", year
    all_years[year] = {}
    top = "nyt/"+str(year)+"/"
    fnames = get_all_files(top)
    for f in fnames:
      all_years[year][f] = parse_lines(f)
  return all_years

def get_tag_list():
  articles = pkl_load("dict_all")
  tag_list = [(tag, []) for tag in tags]
  for k in articles:
    for (tag, lst) in tag_list:
      if tag in articles[k][TAX]:
        lst.append(k)
  return tag_list

def get_vocabs(tag_list):
  for (tag, lst) in tag_list:
    vocab = { UNK : 0 }
    print tag
    for i in range(len(lst)):
      if i%1000 == 0:
        print i, "/", len(lst), len(vocab)/1000
      f = lst[i]
      tokens, label = get_info(f)
      for t in tokens:
        vocab[t] = vocab.get(t, 0) + 1
    pkl_dump(vocab, "vocab_"+tag_shorts[tag])

def truncate_vocabs(cutoff):
  for tag in tag_shorts.values():
    print tag
    d = pkl_load("vocab_"+tag)
    pruned_keys = [UNK] + [k for k in d.keys() if d[k] > cutoff]
    dp = { pruned_keys[i] : i for i in range(len(pruned_keys)) }
    print len(d), "->", len(dp)
    pkl_dump(dp, "vocab_gt"+str(cutoff)+"_"+tag)

def tokenize(text):
  return re.findall(r"[.,!?;]|[\w']+", text)

def write_corenlp_files(text = True):
  for tag in ["US", "world", "sports", "business"]:
    name_list = open("corenlp/"+tag+"_files", "w")
    d_list = pkl_load("text_list_"+tag)
    for d in d_list:
      name = replace(d["fname"], "/", "_")[:-4]
      name_list.write("/Users/bennye/Desktop/thesis/corenlp/"+tag+"/"+name)
      name_list.write("\n")

      if text:
        text_file = open("corenlp/"+tag+"/"+name, "w")
        text_file.write(d["body"])
        text_file.close()

    name_list.close()

def get_cc_deps(f):
  fd = open(f, "r")
  text = fd.read()
  fd.close()
  
  key = "collapsed-ccprocessed-dependencies"

  while True:
    i = text.find(key)
    if i < 0:
      break
    text = text[i+len(key)+2:]
    deps_text = text[:text.find("</dependencies>")-1]
    deps = [l.strip() for l in deps_text.split("\r")][1:-1]
    for j in range(0, len(deps), 4):
      l = deps[j+0]
      d_type = l[l.index("=\"")+2:l.index("\">")]

      l = deps[j+1]
      gov = l[l.index(">")+1:l.index("</")]

      l = deps[j+2]
      dep = l[l.index(">")+1:l.index("</")]

      print d_type, gov, dep
    break

def update_globals(g):
  g["l"] = pkl_load("text_list_world")
  descs = {}
  for d in g["l"]:
    for desc in d["descriptor"]:
      dict_inc(descs, desc)
  vocab = [k for k in descs.keys() if descs[k] > 5]
  g["vocab"] = vocab

def run(g):
  vc = g["vocab"]


def quit(g):
  exit(0)

def main():
  pass

if __name__ == "__main__":
  main()
