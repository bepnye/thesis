def parse_lines(f):
  fd = open(f, "r")
  lines = fd.readlines()

  tax = []
  gen = []
  smm = False
  wrd = 0
  for l in lines:
    if "type=\"taxonomic_classifier\"" in l:
      tax.append(l[l.index(">")+1:l.index("</")])
    if "type=\"general_descriptor\"" in l:
      gen.append(l[l.index(">")+1:l.index("</")])
    if "<abstract>" in l:
      smm = True
    if "item-length=" in l:
      wrd = int(l[l.index("item-length=")+13:l.index(" name=")-1])
  
  fd.close()
  
  return tax, gen, smm, wrd


def get_info(f):
  fd = open(f, "r")
  text = fd.read()
  if "<body.content>" in text:
    body = text[text.index("<body.content>")+14:text.index("</body.content>")]
    tokens = tokenize(body)
  else:
    tokens = []
  return tokens, "<abstract>" in text

def build_text_dicts(tag_list):
  for (tag, short, lst) in tag_list:
    d = []
    for f in lst:
      fd = open(f, "r")
      text = fd.read()
      fd.close()
      lines = text.split("\n")

      cur = { "fname" : f,
              "tag" : short,
              "body": "",
              "abstract": "",
              "descriptor": [],
              "types": [] }

      for l in lines:
        if "</head>" in l:
          break
        if "class=\"indexing_service\" type=\"descriptor" in l:
          cur["descriptor"].append(l[l.index(">")+1:l.index("</")])
        if "types_of_material" in l:
          cur["types"].append(l[l.index(">")+1:l.index("</")])

      if "<abstract>" in text:
        s = text[text.index("<abstract>")+len("<abstract>"):text.index("</abstract>")]
        s = replace(s, "<p>", "")
        s = replace(s, "</p>", "")
        cur["abstract"] = s
        if "</" in s:
          print s

      if "<block class=\"full_text\">" in text:
        body = text[text.index("<block class=\"full_text\">")+len("<block class=\"full_text\">"):]
        s = body[:body.index("</block>")]
        s = replace(s, "<p>", "")
        s = replace(s, "</p>", "")
        cur["body"] = s
        if "</" in s:
          print s

      d.append(cur)
     
    print len(d)
    pkl_dump(d, "text_list_"+short)
