# code uses PEP8 naming conventions
import sys
import math
import subprocess
from tabulate import tabulate
import ntpath
import utility as ut

# TODO REQ Beim Levenstein-Distanz fehlt noch auszugeben, welche Einfügungen,
# Auslassungen und Ersetzungen der Programm in einem beliebig wählbaren Satz
# vorgenommen hat, um eine minimale Distanz zu erhalten.
# function to compute the levenshtein distance
def levenshtein_distance(string1, string2, output):
    """returns the levenstein distance"""

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

    if output == True:
        moves = []
        y = len(str2)
        x = len(str1)
        while x >= 0 and y >= 0:
            if y == 0 and x == 0:
                break
            if y == 0 and x != 0:
                x -= 1
                moves.insert(0, "Insertion")
                continue
            if x == 0 and y != 0:
                moves.insert(0, "Deletion")
                y -= 1
                continue
            if (
                workmatrix[x][y] == workmatrix[x - 1][y - 1]
                and str1[x - 1] == str2[y - 1]
            ):
                moves.insert(0, "Match")
                x -= 1
                y -= 1
            elif (
                workmatrix[x][y] == workmatrix[x - 1][y - 1] + 1
                and workmatrix[x - 1][y - 1] < workmatrix[x][y - 1]
                and workmatrix[x - 1][y - 1] < workmatrix[x - 1][y]
            ):
                moves.insert(0, "Substitution")
                x -= 1
                y -= 1
            elif workmatrix[x][y] == workmatrix[x - 1][y] + 1:
                moves.insert(0, "Insertion")
                x -= 1
            elif workmatrix[x][y] == workmatrix[x][y - 1] + 1:
                moves.insert(0, "Deletion")
                y -= 1
        print(moves)

    return workmatrix[len(str1)][len(str2)]  # value of final cell


# return the number of n grams
def num_n_grams(reference, hypothesis):
    """return the number of n grams"""
    # split reference at n_gram occurrence
    ref = (" " + reference + " ").split(" " + hypothesis + " ")
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
def bleu_num_denom(reference, hypothesis, n):
    """helps calculate the numerator (zaehler) and denominator (nenner) for the bleu metric"""
    hypo_word = hypothesis.split()
    n_gram_list = []
    sum_numerator = 0
    sum_denominator = 0

    # pass all n-grams into the n-gram list
    for x in range(len(hypo_word) - n + 1):
        # TODO check suggestion: add elem = new_element then append if not duplicate
        elem = [hypo_word[i] for i in range(x, x + n)]
        n_gram_list.append(elem) if elem not in n_gram_list else n_gram_list

    # calculate the denominator and numerator then return
    for x in n_gram_list:
        sum_numerator += min(
            num_n_grams(reference, " ".join(x)), num_n_grams(hypothesis, " ".join(x))
        )
        sum_denominator += num_n_grams(hypothesis, " ".join(x))

    return (sum_numerator, sum_denominator)


# function to calculate the modified n-gram precision
def precision(reference, hypothese, n):
    """calculates the modified n-gram precision"""
    if len(reference) != len(hypothese):
        raise Exception("Error: hypothesis and reference files have different lengths!")
    numerator = 0
    denom = 0

    # calculate the numerator and denominator to compute the precision
    for x in range(len(reference)):
        fract = bleu_num_denom(reference[x], hypothese[x], n)
        numerator += fract[0]
        denom += fract[1]

    return numerator / denom


# this function calculates the brevity penalty
def brevity_penalty(referenz, hypothese):
    """calculates the brevity penalty"""
    if len(hypothese) > len(referenz):
        return 1
    else:
        return math.exp(1 - (len(referenz) / len(hypothese)))


# function to compute the bleu metric
# Bilingual Evaluation Study
# Basic idea: How many blocks are identical in reference and hypothesis?
def met_bleu(datei1, datei2, n):
    """computes the bleu metric"""
    referenz = ut.read_from_file(datei1)
    hypothese = ut.read_from_file(datei2)
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

    return bp * math.exp(res)


# function to compute the Word-Error-Rate aka WER
def met_wer(datei1, datei2):
    """compute the Word-Error-Rate aka WER"""
    referenz = ut.read_from_file(datei1)
    hypothese = ut.read_from_file(datei2)
    sum_leven_d = 0

    # compute the sum over all reference-hypothesis pair
    for x in range(len(referenz)):
        sum_leven_d += levenshtein_distance(referenz[x], hypothese[x], False)

    return sum_leven_d / len(" ".join(referenz).split())


# function to compute the Position-independent Error Rate aka PER
def met_per(datei1, datei2):
    """compute the Position-independent Error Rate"""
    referenz = "".join(ut.read_from_file(datei1)).split()
    hypothese = "".join(ut.read_from_file(datei2)).split()
    words = removeduplicate(referenz)
    parity = 0

    # get number of positional accordance / parity
    for word in words:
        parity += hypothese.count(word)

    # return the error rate
    return 1 - ((parity - max(len(hypothese) - len(referenz), 0)) / len(referenz))


def value_counter():
    """uses bash lines to get values from files"""
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


def get_word_len_avr(f_name):
    """given a file get average number of words in each line"""
    cmd = "wc " + f_name + " | awk '{print $2/$1}'"
    cmd = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return float(cmd.communicate()[0][:-2].replace(",", "."))


def main():
    """main() method"""
    # calculate first assignment
    value_counter()
    # calculate levenstein distance
    print(
        "L-Distance: ",
        levenshtein_distance("uz sa a n a n a s sasd awq", "n a n", True),
    )

    col_width = max(len(word) for word in sys.argv) + 2
    table_matrix = [[]]
    table_matrix.append(["FILE", "|", "PER", "|", "WER", "|", "BLEU"])

    # construct the printable table
    for file in sys.argv[1:]:
        table_matrix.append(
            [
                file,
                "|",
                "{:0.4f}".format(
                    met_per("data_exercise_1/newstest.en", "data_exercise_1/" + file)
                ),  # PER
                "|",
                "{:0.4f}".format(
                    met_wer("data_exercise_1/newstest.en", "data_exercise_1/" + file)
                ),  # WER
                "|",
                "{:0.4f}".format(
                    met_bleu(
                        "data_exercise_1/newstest.en", "data_exercise_1/" + file, 4
                    )
                ),  # BLEU
            ]
        )

    print(table_matrix)
    for row in table_matrix:
        print("".join(word.ljust(col_width) for word in row))


def get_path_leaf(path):
    """ For a path return the name of leaf file/folder """
    head, leaf = ntpath.split(path)
    return leaf or ntpath.basename(head)


def compare_bleu_scores(origin, args):
    """
    takes multiple text files as arguments and prints out the BLEU score of each
    one by comparing it to the original text in origin.
    """
    table = []
    headers = ["File Name", "BLEU"]
    for tfile in args:
        # add name and score to table
        table.append(
            [get_path_leaf(tfile), "{:0.4f}".format(met_bleu(origin, tfile, n=4))]
        )

    # print table
    table = tabulate(table, headers=headers, tablefmt="orgtbl")
    print(table)


if __name__ == "__main__":
    main()
    # bleu_num_denom("the cat is on the mat ", "the the cat", 1)
