import scratch, sys

def main():
  g = {}
  while True:
    c = raw_input("cmd: ")

    if c == "q":
      return

    try:
      reload(scratch)
      if c == "g":
        f = scratch.update_globals
      else:
        f = scratch.run
    except:
      print "CATASTROPHIC ERROR:\n\t", sys.exc_info()[1]

    try:
      f(g)
    except:
      print "ERROR:\n\t", sys.exc_info()[1]

main()
