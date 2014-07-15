from utils import *
from matrix import *
from files_to_structs import *
from collections import defaultdict
from subprocess import call
from random import randint
from matplotlib import pyplot
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from gensim import corpora, models, similarities
from scipy.spatial.distance import pdist
import scipy.stats as stats
import scipy.cluster as cluster
from sklearn import linear_model 

STOPWORDS = {w: True for w in stopwords.words('english')}

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
  return [e for e in re.findall(r"[.,!?;]|[\w']+", text) if not STOPWORDS.get(e.lower())]

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
  if not os.path.isfile(f):
    return []
    
  fd = open(f, "r")
  text = fd.read()
  fd.close()
  
  key = "collapsed-ccprocessed-dependencies\">"

  dep_list = []

  while True:
    i = text.find(key)
    if i < 0:
      break
    text = text[i+len(key):]
    deps_text = text[:text.find("</dependencies>")-1]
    deps = [l.strip() for l in deps_text.split("\r")][1:-1]
    for j in range(0, len(deps), 4):
      l = deps[j+0]
      d_type = l[l.index("=\"")+2:l.index("\">")]

      l = deps[j+1]
      gov = l[l.index(">")+1:l.index("</")]

      l = deps[j+2]
      dep = l[l.index(">")+1:l.index("</")]

      dep_list.append([d_type, gov, dep])

  return dep_list

def extract_substring(text, start, stop):
  idx_start = text.find(start)
  if idx_start == -1:
    return -1
  idx_start += len(start)
  return text[idx_start : text.find(stop, idx_start)]

def update_lm(lm, text):
  for t in tokenize(text):
    lm[t] += 1

def dd_0():
  return { key : defaultdict(int) for key in ['word', 'pos', 'ner'] }
def dd_1():
  return defaultdict(int)

def get_pos_etc(fname):
  out = []

  fd = open(fname, 'r')
  lines = fd.readlines()
  fd.close()

  i = 0
  while i < len(lines):
    l = lines[i].strip(); i += 1
    if l[:9] == '<token id':
      l = lines[i].strip(); i += 1
      word  = l[l.index('>')+1:l.index('</')]; l = lines[i].strip(); i += 1
      lemma = l[l.index('>')+1:l.index('</')]; l = lines[i].strip(); i += 1
      start = l[l.index('>')+1:l.index('</')]; l = lines[i].strip(); i += 1
      stop  = l[l.index('>')+1:l.index('</')]; l = lines[i].strip(); i += 1
      pos   = l[l.index('>')+1:l.index('</')]; l = lines[i].strip(); i += 1
      ner   = l[l.index('>')+1:l.index('</')]; l = lines[i].strip(); i += 1
    
      out.append((word, lemma, pos)) 

  return out
 
def get_global_pos_dict(g):
  tag = 'world'
  files = get_all_files('corenlp/'+tag+'_out/')
  pos_dict = defaultdict(dd_1)
  for f in files:
    lst = get_pos_etc(f)
    for (w, pos) in lst:
      pos_dict[w][pos] += 1
  g['world_pos'] = pos_dict

def collect_pos_dicts(g):
  tag = 'world'
  files_abst = get_all_files('corenlp/abst/'+tag+'_out/')
  files_body = set(get_all_files('corenlp/'+tag+'_out/'))
  pairs_list = []
  
  for (i, f_abst) in enumerate(files_abst):
    if i%200 == 0:
      print i

    f_body = 'corenlp/'+tag+'_out/'+ f_abst[f_abst.index('nyt'):]
    if f_body in files_body:
      abst = get_pos_etc(f_abst)
      body = get_pos_etc(f_body)

      pairs_list.append((abst, body))
    else:
      f_orig = f_abst[f_abst.index('nyt'):].replace('_', '/')
      print 'unable to find matched parse for', f_orig

      #fd_orig = open(f_orig, 'r')
      #text = fd_orig.read()
      #fd_orig.close()

      #if "<block class=\"full_text\">" in text:
      #  body = text[text.index("<block class=\"full_text\">")+len("<block class=\"full_text\">"):]
      #  clean_body = body[:body.index("</block>")]
      #  clean_body = replace(clean_body, "<p>", "")
      #  clean_body = replace(clean_body, "</p>", "")
      #  
      #  fd_missing.write('/Users/bennye/Desktop/thesis/corenlp/'+tag+'_new/'+f_orig.replace('/', '_')+'\n')

      #  fd_corenlp = open('corenlp/'+tag+'_new/'+f_orig.replace('/', '_'), 'w')
      #  fd_corenlp.write(clean_body)
      #  fd_corenlp.close()

      #else:
      #  print 'unable to find body for', f_orig

  g[tag+'_pairs_list'] = pairs_list
  pkl_dump(pairs_list, tag+'_pairs_list')

def topic_models(g):
  abstracts = []
  bodies = []

  pairs = g['world_pairs']
  test  = pairs[:len(pairs)/10]
  train = pairs[len(pairs)/10:]
  
  print 'compiling tokens...'
  for [abstract_tokens, body_tokens] in train:
    abstracts.append(abstract_tokens)
    bodies.append(body_tokens)
  
  if 0:
    print 'building vocab...'
    bow_dict = corpora.Dictionary(bodies)
    g['bow_dict'] = bow_dict
    
    print 'constructing corpus...'
    bow_corpus = [bow_dict.doc2bow(text) for text in bodies]
    g['bow_corpus'] = bow_corpus
    bow_corpus_abst = [bow_dict.doc2bow(text) for text in abstracts]
    g['bow_corpus_abst'] = bow_corpus_abst

    print 'training LDA...'
    lda = models.ldamodel.LdaModel(bow_corpus, id2word=bow_dict, num_topics=30, chunksize = 100, passes = 1, update_every = 1)
    g['lda'] = lda

    print 'building X...'
    X = []
    for (i, doc) in enumerate(bow_corpus):
      if i % 300 == 0:
        print i, '/', len(bow_corpus)
      v = [0]*lda.num_topics
      for (topic, value) in lda[doc]:
        v[topic] = value
      # gotta get the constant
      X.append(v)
    g['X'] = X

    print 'building Y...'
    Y = []
    for (i, doc) in enumerate(bow_corpus_abst):
      if i % 300 == 0:
        print i, '/', len(bow_corpus_abst)
      v = [0]*lda.num_topics
      for (topic, value) in lda[doc]:
        v[topic] = value
      Y.append(v)
    g['Y'] = Y

  else:
    print 'loading from g...'
    bow_dict = g['bow_dict']
    bow_corpus = g['bow_corpus']
    bow_corpus_abst = g['bow_corpus_abst']
    lda = g['lda']
    X = g['X']
    Y = g['Y']

  X_t = zip(*X)
  Y_t = zip(*Y)

  for i in range(lda.num_topics):
    print np.mean(X_t[i]), np.mean(Y_t[i])

  pprint = lambda v: ''.join(['\033['+['41', '43', '42'][0 if x < -0.2 else (1 if x < 0.2 else 2)]+'m' + ' '*bool(x>0)+str(round(x,2))[2:].ljust(3-bool(x>0), '0') + '\033[0m' for x in v])
  
  Yp_t = zip(*g['Yp'])
  Xp = g['Xp']
  for i in range(lda.num_topics):
    reg = linear_model.LinearRegression()
    reg.fit(X, Y_t[i])
    #print pprint(reg.coef_)
    print np.mean((reg.predict(Xp) - Yp_t[i]) ** 2)


def scratch(g):
  abst_lm = defaultdict(lambda : defaultdict(int))
  body_lm = defaultdict(lambda : defaultdict(int))
  only_body_lm = defaultdict(lambda : defaultdict(int))

  pos_dict = g['world_pos']
  for (fname, d) in g['world_pairs'].items():
    abstract_tokens = tokenize(d['abstract'].replace('<p>', '').replace('</p>', ''))
    body_tokens = tokenize(d['body'].replace('<p>', '').replace('</p>', ''))
    abstract_set = set(abstract_tokens)
    only_body_tokens = [t for t in body_tokens if t not in abstract_set]

    for token in abstract_tokens:
      if token in pos_dict:
        pos = max(pos_dict[token].items(), key = lambda x: x[1])[0][:2]
        abst_lm[pos][token] += 1

    for token in body_tokens:
      if token in pos_dict:
        pos = max(pos_dict[token].items(), key = lambda x: x[1])[0][:2]
        body_lm[pos][token] += 1

    for token in only_body_tokens:
      if token in pos_dict:
        pos = max(pos_dict[token].items(), key = lambda x: x[1])[0][:2]
        only_body_lm[pos][token] += 1


  g['abst_lm'] = abst_lm
  g['body_lm'] = body_lm
  g['only_body_lm'] = only_body_lm

def build_pairs(g):
  l = []
  for (fname, d) in g['taglist']['Top/News/World'].items():
    if d['abstract']:
      abstract_tokens = tokenize(d['abstract'].replace('<p>', '').replace('</p>', ''))
      body_tokens = tokenize(d['body'].replace('<p>', '').replace('</p>', ''))
      l.append([abstract_tokens, body_tokens])

  print l[0], len(l)
  g['world_pairs'] = l
  pkl_dump(l, 'world_pairs')
  return

def norm_lm(d):
  N = float(sum([d[k] for k in d.keys()]))
  return { k : d[k]/N for k in d.keys() }

def make_clusters_fresh(g, target_pos = 'VB'):

  print 'transposing...'
  absts, bodys = zip(*g['world_pairs_list'])

  print 'extracting', target_pos+'...'
  bodys_pos = []
  absts_pos = []
  
  for doc in bodys:
    cur_vec = []
    for i in range(len(doc)-1):
      if doc[i][2] == target_pos or doc[i+1][2] == target_pos:
        cur_vec.append(doc[i][1]+'_'+doc[i+1][1])
    bodys_pos.append(cur_vec)

  for doc in absts:
    for i in range(len(doc)-1):
      if doc[i][2] == target_pos or doc[i+1] == target_pos:
        cur_vec.append(doc[i][1]+'_'+doc[i+1][1])
    absts_pos.append(cur_vec)
  
  print 'building POS vocabs...'
  bodys_vocab = set({})
  for doc in bodys_pos:
    for lemma in doc:
      bodys_vocab.add(lemma)
  
  absts_vocab = set({})
  for doc in absts_pos:
    for lemma in doc:
      absts_vocab.add(lemma)

  print 'extracting all...'
  bodys_all = []
  absts_all = []

  for doc in bodys:
    cur_vec = []
    for i in range(len(doc)-1):
      cur_vec.append(doc[i][1]+'_'+doc[i+1][1])
    bodys_all.append(cur_vec)

  for doc in absts:
    cur_vec = []
    for i in range(len(doc)-1):
      cur_vec.append(doc[i][1]+'_'+doc[i+1][1])
    absts_all.append(cur_vec)

  print "training w2v..."
  w2v = models.word2vec.Word2Vec(bodys_all, min_count = 2, workers = 3, size = 100)

  print 'building X...'
  vocab, X = zip(*[(w, w2v[w]) for w in bodys_vocab if w in w2v])
  
  g['bodys_pos']   = bodys_pos
  g['bodys_vocab'] = bodys_vocab
  g['bodys_all']   = bodys_all

  g['absts_pos']   = absts_pos
  g['absts_vocab'] = absts_vocab
  g['absts_all']   = absts_all

  g['w2v']         = w2v
  g['vocab']       = vocab
  g['X']           = X

  g['target_pos']  = target_pos

  make_clusters_old(g)
 
def make_clusters_old(g):

  w2v        = g['w2v']
  vocab      = g['vocab']
  X          = g['X']
  target_pos = g['target_pos']


  hierarchy = True
  kmeans    = False

  if hierarchy:
    print 'computing dist mat...'
    dist_mat = pdist(X)

    #if 0:
    #  print 'getting ssets...'
    #  vocab_ssets = []
    #  for w in vocab:
    #    ssets = wn.synsets(w, pos = 'v' if target_pos == 'VB' else 'n')
    #    if len(ssets) > 0:
    #      vocab_ssets.append([w, ssets[0]])

    #  print 'finding similiarities...'
    #  N = len(vocab_ssets)
    #  dist_mat = np.empty([N,N])

    #  for (i, [w_i, sset_i]) in enumerate(vocab_ssets):
    #    print i
    #    for (j, [w_j, sset_j]) in enumerate(vocab_ssets):
    #      dist = sset_i.path_similarity(sset_j)
    #      dist_mat[i][j] = dist

    print 'doing linkage...'
    linkage_mat = cluster.hierarchy.linkage(dist_mat, 'complete')
    g['linkage_mat'] = linkage_mat

    clusters = { i: [w] for (i, w) in enumerate(vocab) }
    for (i, [idx1, idx2, dist, count]) in enumerate(linkage_mat):
      print 'Merged', clusters[int(idx1)], 'and', clusters[int(idx2)] 
      clusters[i+len(vocab)] = clusters[int(idx1)] + clusters[int(idx2)]

    return clusters
    #word_clusters = defaultdict(list)
    #for i in range(len(vocab)):
    #  word_clusters[clusters[i]].append(vocab[i])
    #for k in word_clusters:
    #  print word_clusters[k]
    #g['world_clusters_abst_'+pos] = word_clusters

  if kmeans:
    print 'whitening data...'
    white = cluster.vq.whiten(X)
    print 'computing clusters...'
    book, distortion = cluster.vq.kmeans(white, len(vocab)/10, iter = 2)
    labels, _ = cluster.vq.vq(white, book)
    word_clusters = defaultdict(list)
    for i in range(len(vocab)):
      word_clusters[labels[i]].append(vocab[i])
    for k in word_clusters:
      print word_clusters[k]
    g['world_clusters_abst_'+pos] = word_clusters


def load(g, name):
  g[name] = pkl_load(name)

def run(g):

  vocab = g['vocab']
  for w in vocab:
    lemmas = sum([synset.lemmas for synset in wn.synsets(w)], [])
    print w, set([lemma.name for lemma in lemmas])
  return

  absts, bodys = zip(*g['world_pairs_list'])
  print 'extracting VBs...'
  absts_pos = [[lemma for (word, lemma, pos) in doc if pos[:2] == 'VB'] for doc in absts]
  bodys_pos = [[lemma for (word, lemma, pos) in doc if pos[:2] == 'VB'] for doc in bodys]

  absts_2 = [[tuple(doc[i:i+2]) for i in range(len(doc)-1)] for doc in absts_pos]
  bodys_2 = [[tuple(doc[i:i+2]) for i in range(len(doc)-1)] for doc in bodys_pos]
  
  abst_counter = defaultdict(int)
  body_counter = defaultdict(int)

  print 'building counters...'
  for doc in absts_2:
    for w in doc:
      abst_counter[w] += 1
  for doc in bodys_2:
    for w in doc:
      body_counter[w] += 1

  abst_n = float(sum(abst_counter.values()))
  body_n = float(sum(body_counter.values()))

  for w in abst_counter:
    abst_counter[w] /= abst_n
  for w in body_counter:
    body_counter[w] /= body_n

  diff_counter = { k : abst_counter[k] - body_counter[k] for k in set(abst_counter.keys()).union(body_counter.keys()) }
  
  sorted_diff = sorted(diff_counter.items(), key = lambda e: -1*e[1])

  max_w_len = 9
  print 'biased towards abst'
  for (w, diff) in sorted_diff[:15]:
    print w, ' '*(max_w_len - len(w)), round(diff, 4)
  print 'biased towards body'
  for (w, diff) in sorted_diff[-15:][::-1]:
    print w, ' '*(max_w_len - len(w)), -1*round(diff, 4)

def init(g):
  g['world_pairs_list'] = pkl_load('world_pairs_list')

def quit(g):
  exit(0)

def main():
  pass

if __name__ == "__main__":
  main()
