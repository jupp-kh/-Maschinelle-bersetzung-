# code uses PEP8 naming conventions
import sys
import math
import os
import subprocess
from math import pi


def get_minimum_of_line(workmatrix, line):
    min_val = workmatrix[line][1]
    pos = 1
    for y in range(len(workmatrix[line])):
        if workmatrix[line][y] <= min_val:
            min_val = workmatrix[line][y]
            pos = y
    return pos


# TODO REQ Beim Levenstein-Distanz fehlt noch auszugeben, welche Einfügungen,
# Auslassungen und Ersetzungen der Programm in einem beliebig wählbaren Satz
# vorgenommen hat, um eine minimale Distanz zu erhalten.
# function to compute the levenshtein distance
def levenshtein_distance(string1, string2, output):
    str1 = string1.split(" ")
    str2 = string2.split(" ")
    # initialise matrix structure
    workmatrix = [[0 for x in range(len(str2) + 1)] for x in range(len(str1) + 1)]
    workmatrix[0][0] = 0
    for i in range(0, len(str1) + 1):
        workmatrix[i][0] = i
    workmatrix[0][:] = range(0, len(str2) + 1)

    # calculates the distance in the final cell of the matrix
    for x in range(1, len(str1) + 1):
        for y in range(1, len(str2) + 1):
            if str1[x - 1] == str2[y - 1]:  # preserve value if sequences are aligned
                workmatrix[x][y] = workmatrix[x - 1][y - 1]
            else:  # get minimum of (substitution , insertion, deletion)
                value = min(
                    workmatrix[x - 1][y - 1],
                    workmatrix[x][y - 1],
                    workmatrix[x - 1][y],
                )
                workmatrix[x][y] = value + 1

    deletions = 0
    insertion = 0
    operations = ""
    last_min = 0

    if output:
        for x in range(1, len(str1) + 1):
            min_pos = get_minimum_of_line(workmatrix, x)
            op = ""
            if x != 1:
                if min_pos - last_min >= 1:
                    min_pos = last_min + 1
            if workmatrix[x - 1][min_pos - 1] == workmatrix[x][min_pos]:
                op = "Match "
            else:
                op = "Substitution "
            if min_pos - deletions + insertion > x and x == 1:
                deletions += 1
                operations += "Deletion " + op
            elif min_pos - deletions + insertion < x and min_pos == len(str2):
                insertion += 1
                operations += "Insertion "
            elif min_pos - deletions + insertion == x:
                operations += op
            last_min = min_pos
        print(operations)
    for row in workmatrix:
        print(row)
    return workmatrix[len(str1)][len(str2)]  # value of final cell


# return the number of n grams
def num_n_grams(referenz, hypothese):
    ref = referenz.split(hypothese)
    return len(ref) - 1


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
def bleu_num_denom(referenz, hypothese, n):
    hypo_word = hypothese.split()
    n_gram_list = []
    sumzaehler = 0
    sumnenner = 0

    # pass all n-grams into the n-gram list
    for x in range(len(hypo_word) - n + 1):
        # TODO check suggestion: add elem = new_element then append if not duplicate
        elem = [hypo_word[i] for i in range(x, x + n)]
        n_gram_list.append(elem) if elem not in n_gram_list else n_gram_list

    # clear duplicates in list
    n_gram_list = removeduplicate(n_gram_list)

    # calculate the denominator and numerator then return
    for x in n_gram_list:
        sumzaehler += min(
            num_n_grams(referenz, " ".join(x)), num_n_grams(hypothese, " ".join(x))
        )
        sumnenner += num_n_grams(hypothese, " ".join(x))
    return (sumzaehler, sumnenner)


# this method reads from file datei
# returns a list of read file lines
def read_from_file(datei):
    with open(datei, "r", encoding="utf-8") as in_file:
        res = []

        # append list res with formatted lines
        for line in in_file:
            res.append(line.strip())

    return res


# function to calculate the modified n-gram precision
def precision(reference, hypothese, n):
    if len(reference) != len(hypothese):
        raise Exception("Error: hypothesis and reference files have different lengths!")
    zaehler = 0
    nenner = 0

    # calculate the numerator and denominator to compute the precision
    for x in range(len(reference)):
        fract = bleu_num_denom(reference[x], hypothese[x], n)
        zaehler += fract[0]
        nenner += fract[1]

    return zaehler / nenner


# this function calculates the brevity penalty
def brevity_penalty(referenz, hypothese):
    if len(hypothese) > len(referenz):
        return 1
    else:
        return math.exp(1 - (len(referenz) / len(hypothese)))


# function to compute the bleu metric
# Bilingual Evaluation Study
# Basic idea: How many blocks are identical in reference and hypothesis?
def met_bleu(datei1, datei2, n):
    referenz = read_from_file(datei1)
    hypothese = read_from_file(datei2)
    res = 0

    # calculate the precision
    # calculate the
    for x in range(1, n + 1):
        # print (x)
        temp = precision(referenz, hypothese, x)
        if temp != 0:
            res += 1 / n * math.log(temp)

    # get the brevity penalty
    bp = brevity_penalty(" ".join(referenz).split(), " ".join(hypothese).split())

    # print("breveity penalty: ", bp, res)
    return bp * math.exp(res)


# function to compute the Word-Error-Rate aka WER
def met_wer(datei1, datei2):
    referenz = read_from_file(datei1)
    hypothese = read_from_file(datei2)
    sum_leven_d = 0

    # compute the sum over all reference-hypothesis pair
    for x in range(len(referenz)):
        sum_leven_d += levenshtein_distance(referenz[x], hypothese[x], False)

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
    return 1 - ((parity - max(len(hypothese) - len(referenz), 0)) / len(referenz))


def value_counter():
    # Berechnen Sie folgende Korpus-Statistiken für Quell- und Zielseite der Testdaten
    # (Dateien newstest.de und newstest.en): Anzahl der laufenden Wörter, Anzahl verschiedener
    # Wörter, durchschnittliche Satzlänge. Setzen Sie dazu Unix-Tools wie sed und wc ein.
    # Anmerkung: Unter „laufenden Wörter“ ist die Gesamtanzahl an Wörtern gemeint, wobei
    # mehrfach Vorkommnisse eines gleichen Wortes auch mehrfach gezählt werden.
    print("Word count of each file:")  # Gesamtanzahl an Wörtern
    for file in sys.argv[1:]:
        # count number of words
        # wc -w file
        subprocess.run(["wc", "-w", file])

        # count number of words - no duplicates
        get_num_words = "grep -wo '[[:alnum:]]\+' " + file + "| sort | uniq -c | wc -l"

        # calculate the average length of sentences
        get_avr_len = (
            "awk '{total=length($0)+total; len=len+1;} END {print total/len}' " + file
        )

        # call subprocesses - bash
        grep = subprocess.Popen(
            get_num_words,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        awk = subprocess.Popen(
            get_avr_len,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

        # print outputs of each command
        print(
            " Number of different words",
            grep.communicate()[0],
            "Average Line Length:",
            awk.communicate()[0],
        )


def main():
    # calculate first assignment
    # value_counter()
    # calculate levenstein distance
    print(
        "L-Distance: ",
        levenshtein_distance("b a n a n e", "a n a n a s", True),
    )

    # col_width = max(len(word) for word in sys.argv) + 2
    # table_matrix = [[]]
    # table_matrix.append(["FILE", "|", "PER", "|", "WER", "|", "BLEU"])

    # # construct the printable table
    # for file in sys.argv[1:]:
    #     table_matrix.append(
    #         [
    #             file,
    #             "|",
    #             "{:0.4f}".format(met_per("newstest.en", file)),  # PER
    #             "|",
    #             "{:0.4f}".format(met_wer("newstest.en", file)),  # WER
    #             "|",
    #             "{:0.4f}".format(met_bleu("newstest.en", file, 4)),  # BLEU
    #         ]
    #     )

    # print(table_matrix)
    # for row in table_matrix:
    #     print("".join(word.ljust(col_width) for word in row))


if __name__ == "__main__":
    main()
