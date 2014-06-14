from utils import *

def build_X_Y(fnames, vocab):
  rows = []
  cols = []
  vals = []

  labels = []

  for i in range(len(fnames)):
    f = fnames[i]
    tokens, label = get_info(f)
    tokens = [t if vocab.get(t, False) else UNK for t in tokens]
    t_c = Counter(tokens)
    labels.append(int(label))
    for t in t_c:
      rows.append(i)
      cols.append(vocab[t])
      vals.append(t_c[t])
    if i%1000 == 0:
      print i, "/", len(fnames)

  return sparse.csr_matrix((vals, (rows,cols)), shape = (len(fnames), len(vocab))), np.array(labels)

def classify(X, Y, vocab):

  print "slicing..."
  N = np.shape(X)[0]

  test_idx = [] 
  
  for i in range(N/10):
    test_idx.append(randint(0, N-1))

  train_idx = [i for i in range(N) if i not in test_idx]

  X_test = X[test_idx]
  X_train = X[train_idx]
  
  print np.shape(X_test)
  print np.shape(X_train)

  Y_test = Y[test_idx]
  Y_train = Y[train_idx]

  print "training..."
  lr = linear_model.LogisticRegression(class_weight = 'auto')
  lr.fit(X_train, Y_train)

  #vocab_rev = { vocab[k] : k for k in vocab }

  #top_15 = zip(*sorted([(vocab_rev[i], lr.coef_[0][i]) for i in range(len(lr.coef_[0]))], key=lambda x: -1*x[1])[:15])[0]
  #bot_15 = zip(*sorted([(vocab_rev[i], lr.coef_[0][i]) for i in range(len(lr.coef_[0]))], key=lambda x:    x[1])[:15])[0]

  #print top_15
  #print
  #print bot_15

  print "testing..."
  Y_hat = lr.predict(X_test)

  correct = 0.0
  total = 0.0
  for i in range(len(Y_hat)):
    total += 1.0
    if Y_hat[i] == Y_test[i]:
      correct += 1.0
  print correct, "/", total, correct/total
