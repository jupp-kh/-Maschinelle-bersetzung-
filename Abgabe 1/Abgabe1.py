import math
def LevenshteinDistan(string1  , string2):
    str1 = string1.split(" ")
    str2 = string2.split(" ")
    workmatrix = [[0 for x in range(len(str2)+1)] for x in range(len(str1)+1)]
    workmatrix[0][0] = 0
    for x in range(len(str1)+1):
        for y in range(len(str2)+1):
            if x == 0 and y == 0: 
                continue
            elif x == 0: 
                workmatrix[x][y] = workmatrix[x][y-1] + 1
            elif y == 0: 
                workmatrix[x][y] = workmatrix [x-1][y] + 1
            elif str1[x-1] == str2[y-1]:
                workmatrix[x][y] = workmatrix[x-1][y-1]
            else:
                workmatrix[x][y] = min(workmatrix[x-1][y-1],workmatrix[x][y-1],workmatrix[x-1][y]) + 1
                 
    return workmatrix[len(str1) ][len(str2)]

def nGram(referenz,hypothese):
    ref = referenz.split(hypothese)
    return len(ref)-1

def removeduplicate(l):
    res = []
    for x in l: 
        if x not in res:
            res.append(x)
    return res 
   
def helpBleu(referenz,hypothese,n):
    hypo = hypothese.split()
    ngramlist = []
    sumzaehler = 0
    sumnenner = 0
    for x in range(len(hypo)-n+1):
        ngramlist.append([hypo[i] for i in range(x, x+n)])
    ngramlist =  removeduplicate(ngramlist)
    for x in ngramlist:
        sumzaehler +=  min(nGram(referenz, " ".join(x)),nGram(hypothese, " ".join(x)))
        sumnenner += nGram(hypothese," ".join(x))
    return (sumzaehler, sumnenner)

def reader (datei):
    in_file = open(datei, 'r', encoding="utf-8")
    res = []
    for line in in_file:
        res.append(line.strip())
    in_file.close()
    return res

def precision(referenz,hypothese, n): 
    if (len(referenz) != len(hypothese)):
        raise Exception("error: hypothese and referenz datei haben verschiedene länge")
    zaehler = 0 
    nenner = 0
    for x in range (len(referenz)):
        fract = helpBleu(referenz[x],hypothese[x],n)
        zaehler += fract[0]
        nenner += fract [1]
    return zaehler/ nenner

def BrevityPenalty(referenz,hypothese):

    if len(hypothese) > len(referenz):
        return 1
    else:
        return math.exp(1 - (len(referenz)/ len(hypothese)))

def Bleu(datei1, datei2,n): 
    referenz = reader(datei1)
    hypothese = reader(datei2)
    precision(referenz,hypothese,n)
    res = 0

    for x in range(1,n+1):
        print (x)
        temp = precision(referenz,hypothese,x)
        if temp != 0:
            res += (1/n * math.log(temp))
    
    bp = BrevityPenalty(" ".join(referenz).split()," ".join(hypothese).split())
    print(bp, res)
    return bp* math.exp(res)

def wer(datei1, datei2):
    referenz = reader(datei1)
    hypothese = reader(datei2)
    sumlevenstde = 0
    for x in range(len(referenz)):
        sumlevenstde += LevenshteinDistan(referenz[x],hypothese[x])

    return sumlevenstde / len(" ".join(referenz).split())

def per(datei1, datei2):
    referenz = "".join(reader(datei1)).split()
    hypothese = "".join(reader(datei2)).split()
    words = removeduplicate(referenz)
    res = 0
    for word in words:
        res += hypothese.count(word) 
    return 1-((res - max(len(hypothese)-len(referenz),0))/len(referenz))


def printer (m):    
    for row in m: 
        print(row)
#print(LevenshteinDistan("ich dich dich dich dich", "ich dich dich hasse dich"))
#print(LevenshteinDistan("do you love me do you do you ", "do do you need me do you do you "))

print(Bleu("newstest.en","newstest1.hyp3",4))
print(per("newstest.en","newstest1.hyp3"))
print(wer("newstest.en","newstest1.hyp3"))