def manually_label_articles(g):
  tag = "world"
  l = g[tag]["l"]
  g[tag]["res"] = g[tag].get("res", [])
  for i in range(g.get("n_articles", 20)):
    n = randint(0, len(l)-1)
    call("echo \"" + l[n]["body"].replace("        ", "\n")[2:].replace("\"", "''") +"\" | open -f", shell=True)
    rating = input()
    g[tag]["res"].append((rating, bool(l[n]["abstract"])))

  t = [v for (v, tf) in g[tag]["res"] if tf == True]
  f = [v for (v, tf) in g[tag]["res"] if tf == False]
  bins = np.linspace(-1, 6, 7)
  pyplot.hist(t, bins, alpha=0.5, label='with')
  pyplot.hist(f, bins, alpha=0.5, label='without')
  pyplot.legend(loc='upper right')
  pyplot.show()

def date_str_to_int(s, only_month = False):
  year  = int(s[0:4])
  month = int(s[4:6])
  day   = int(s[6:8])

  months = [ 31, 28, 31, 30,
             31, 30, 31, 31,
             30, 31, 30, 31 ]

  if only_month:
    return 12*(year - 2005) + month - 1
  else:
    return 365*(year - 2005) + sum(months[:month-1]) + day

def plot_per_tag_hists(g):
  date_tag = "date.publication="
  tags = ["US", "world", "sports", "business"]
  plots = [411, 412, 413, 414]

  for i in range(4): 
    res = defaultdict(list)
    #res = []
    tag = tags[i]
    for d in g[tag]["l"]:
      fd = open(d["fname"], "r")
      lines = fd.readlines()
      fd.close()
      abst = d["abstract"] == ""
      date = -1
      for l in lines:
        if date_tag in l:
          idx = l.find(date_tag)+len(date_tag)+1
          date_str = l[idx:idx+15]
          date_int = date_str_to_int(date_str, False)
          break
      res[date_int].append(abst)
      #res.append((date_int, abst))
    
    #a = [v for (v, b) in res]
    #t = [v for (v, b) in res if b == True]
    #f = [v for (v, b) in res if b == False]
    
    x = [sum(res[k])/float(len(res[k])) for k in res]
    weights = np.ones_like(x)/len(x)
    
    pyplot.subplot(plots[i])
    bins = np.linspace(0, 12*3, 12*3)
    pyplot.ylabel("Per day density")
    #pyplot.ylabel("# of articles")
    pyplot.hist(x, 25, label = tag, weights = weights)
    #pyplot.hist(t, bins, alpha=0.5, label=tag+', with abstract')
    #pyplot.hist(f, bins, alpha=0.5, label=tag+', without abstract')
    #pyplot.hist(a, bins, alpha=0.5, label=tag+', all')
    pyplot.legend(loc='center')

  pyplot.xlabel("Percentage of articles summarized for a given day")
  #pyplot.xlabel("Month (since Jan 2005)")
  pyplot.show()

def predict_date(g):
  date_tag = "date.publication="
  tags = ["US", "world", "sports", "business"]
  for tag in tags:
    X = []
    Y = []
    for d in g[tag]["l"]:
      fd = open(d["fname"], "r")
      lines = fd.readlines()
      fd.close()
      abst = d["abstract"] == ""
      date = -1
      for l in lines:
        if date_tag in l:
          idx = l.find(date_tag)+len(date_tag)+1
          date_str = l[idx:idx+15]
          date_int = date_str_to_int(date_str, False)
          break
      x = [0]*365*3
      x[date_int] = 1
      X.append(x)
      Y.append(int(abst))

    N = len(Y)

    X = np.array(X)
    Y = np.array(Y)

    X_test  = X[:N/10]
    Y_test  = Y[:N/10]
    
    X_train = X[N/10:]
    Y_train = Y[N/10:]

    lr = linear_model.LogisticRegression()
    lr.fit(X_train, Y_train)
    
    print "testing..."
    Y_hat = lr.predict(X_test)

    correct = 0.0
    total = 0.0
    for i in range(len(Y_hat)):
      total += 1.0
      if Y_hat[i] == Y_test[i]:
        correct += 1.0
    print correct, "/", total, correct/total
