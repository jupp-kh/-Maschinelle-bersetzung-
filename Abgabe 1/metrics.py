# code uses PEP8 naming conventions
import math


# function to compute the levenshtein distance 
def levenshtein_distance(string1, string2):
    str1 = string1.split(" ")
    str2 = string2.split(" ")
    workmatrix = [[0 for x in range(len(str2)+1)] for x in range(len(str1)+1)]
    workmatrix[0][0] = 0

    # iteratively compute the cells of the workmatrix
    # ecaluates the distance in the final cell of the matrix
    for x in range(len(str1)+1):
        for y in range(len(str2)+1):
            if x == 0 and y == 0:   # skip first element of matrix
                continue
            elif x == 0:            # TODO case explanation 
                workmatrix[x][y] = workmatrix[x][y-1] + 1
            elif y == 0:            # TODO case explanation
                workmatrix[x][y] = workmatrix [x-1][y] + 1
            elif str1[x-1] == str2[y-1]:    # TODO case explanation
                workmatrix[x][y] = workmatrix[x-1][y-1]
            else:                   #otherwise
                workmatrix[x][y] = min(workmatrix[x-1][y-1],workmatrix[x][y-1],workmatrix[x-1][y]) + 1
                      
    return workmatrix[len(str1) ][len(str2)] # value of final cell



# return the number of n grams 
def n_gram(referenz,hypothese):
    ref = referenz.split(hypothese)
    return len(ref)-1



# this function removes duplicate elements for a given list lis
def removeduplicate(lis):
    res = []

    # iterate and add non-recurring elements 
    for x in lis: 
        if x not in res: 
            res.append(x)
    return res 



# helps calculate the numerator (zaehler) and denominator (nenner) for the bleu metric 
# returns the tuple (numerator, denominator)
def bleu_num_denom(referenz,hypothese,n):
    hypo_word = hypothese.split()
    n_gram_list = []
    sumzaehler = 0
    sumnenner = 0

    # pass all n-grams into the n-gram list 
    for x in range(len(hypo_word)-n+1):
        # TODO suggestion: add elem = new_element then append if not duplicate
        n_gram_list.append([hypo_word[i] for i in range(x, x+n)])
    
    # clear duplicates in list
    n_gram_list =  removeduplicate(n_gram_list)
    
    # calculate the denominator and numerator then return
    for x in n_gram_list:
        sumzaehler +=  min(n_gram(referenz, " ".join(x)),n_gram(hypothese, " ".join(x)))
        sumnenner += n_gram(hypothese," ".join(x))
    return (sumzaehler, sumnenner)





# this method reads from file datei 
# returns a list of read file lines
def read_from_file(datei):
    with open(datei, 'r', encoding="utf-8") as in_file:
        res = []

        # append list res with formatted lines
        for line in in_file:
            res.append(line.strip())
        
    return res



# function to calculate the modified n-gram precision 
def precision(reference,hypothese, n): 
    if (len(reference) != len(hypothese)):
        raise Exception("Error: hypothesis and reference files have different lengths!")
    zaehler = 0 
    nenner  = 0

    # calculate the numerator and denominator to compute the precision
    for x in range (len(reference)):
        fract = bleu_num_denom(reference[x],hypothese[x],n)
        zaehler += fract[0]
        nenner += fract [1]

    return zaehler / nenner



# this function calculates the brevity penalty
def brevity_penalty(referenz,hypothese):
    if len(hypothese) > len(referenz):
        return 1
    else:
        return math.exp(1 - (len(referenz)/ len(hypothese)))



# function to compute the bleu metric
# Bilingual Evaluation Study
def met_bleu(datei1, datei2,n): 
    referenz = read_from_file(datei1)
    hypothese = read_from_file(datei2)
    precision(referenz,hypothese,n)
    res = 0

    for x in range(1,n+1):
        print (x)
        temp = precision(referenz,hypothese,x)
        if temp != 0:
            res += (1/n * math.log(temp))

    bp = brevity_penalty(" ".join(referenz).split()," ".join(hypothese).split())
    
    print(bp, res)
    return bp * math.exp(res)




# function to compute the Word-Error-Rate aka WER
def met_wer(datei1, datei2):
    referenz = read_from_file(datei1)
    hypothese = read_from_file(datei2)
    sum_leven_d = 0

    # compute the sum over all reference-hypothesis pair
    for x in range(len(referenz)):
        sum_leven_d += levenshtein_distance(referenz[x],hypothese[x])

    return sum_leven_d / len(" ".join(referenz).split())




# function to compute the Position-independent Error Rate aka PER
def met_per(datei1, datei2):
    referenz = "".join(read_from_file(datei1)).split()
    hypothese = "".join(read_from_file(datei2)).split()
    words = removeduplicate(referenz)
    parity = 0

    # get number of positional accordance / parity 
    for word in words:
        parity += hypothese.count(word) 

    # return the error rate
    return 1-((parity - max(len(hypothese)-len(referenz),0))/len(referenz))



# printer function 
# outputs lines of "data" to the console
def printer(data):    # currently unused
    for row in data: 
        print(row)



#print(LevenshteinDistance("ich dich dich dich dich", "ich dich dich hasse dich"))
#print(LevenshteinDistance("do you love me do you do you ", "do do you need me do you do you "))


print(met_bleu("newstest.en", "newstest.hyp3", 4))
print(met_per("newstest.en", "newstest.hyp3"))
print(met_wer("newstest.en","newstest.hyp3"))
