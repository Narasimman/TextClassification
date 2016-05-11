import re

with open('data/TaggedData/response-money-fx.txt', 'r') as content_file:
    content = content_file.read()
content = content.replace("\n","",100000000)
content = content.replace("<sentence>","\n",10000000)
content = content.replace("</sentence>","\n",10000000)
content = content.replace("</constit></token>","\n",100000000)
content = content.split('\n')
for line in content:
	line = re.sub(r'(^[<token intvalue="]+)([\d]+)">',r'',line)
	line = line.replace("<token>","")
	line = line.replace('<token case="cap">',"")
	line = line.replace('<token case="forcedCap">',"")
	line = line.replace("<constit cat=","")
	line = line.replace('">',"\t")
	if (len(line) > 0):
		if (line[0] == '"'):
			line = line.replace('"',"",1)
	print line
