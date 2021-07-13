import sys, time, threading
import os
import csv
import subprocess

cur_dir = os.path.dirname(__file__)


def max_word_in_line(filepath):
    """returns max number of words in line"""
    awk = (
        "awk 'NF > max_length { max_length = NF; longest_line = $0 } END { print max_length }'"
        + str(filepath)
    )
    awk = subprocess.Popen(
        awk,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    return awk.communicate()[0]


# process function for animation
def proc_function():
    n = 20
    for i in range(n):
        time.sleep(1)
        sys.stdout.write(
            "\r"
            + "loading... "
            + str(i)
            + "/"
            + str(n)
            + " "
            + "{:.2f}".format(i / n * 100)
            + "%"
        )
        sys.stdout.flush()
    sys.stdout.write("\r" + "File created         \n")


# function for animated character
def animated_loading():
    chars = "/—\|"
    for char in chars:
        sys.stdout.write("\r" + "loading... " + char)
        time.sleep(0.1)
        sys.stdout.flush()


def loader():
    proc = threading.Thread(name="process", target=proc_function)
    proc.start()
    while proc.isAlive():
        animated_loading()


# this method reads from file datei
# returns a list of read file lines
## FIXME i cant understand deutsch
def read_from_file(datei, start=0, end=-1):
    with open(datei, "r", encoding="utf-8") as in_file:
        res = []
        lines = in_file.readlines()

        if end == -1:
            end = len(lines)

        # append list res with formatted lines
        for i, line in enumerate(lines):
            if i in range(start, end):
                res.append(line.strip())
    return res


def save_as_csv(file_des, data):
    """ Saves data str as csv in file_des """
    try:
        os.remove(file_des)
    except:
        print("No file, creating new file")

    with open(file_des, "x", newline="", encoding="utf-8") as file_csv:
        writer = csv.writer(file_csv)
        for x in data:
            writer.writerow([x])
    file_csv.close()


def save_as_txt(file_des, data):
    """ Saves data str as txt in file_des """
    try:
        os.remove(file_des)
    except:
        print("No file, creating new file")

    with open(file_des, "x", encoding="utf-8") as write_f:
        write_f.write(data)


def save_list_as_txt(file_des, data_list, strings=False):
    """ Saves data_list as csv in file_des """
    try:
        os.remove(file_des)
    except:
        print("No file, creating new file")
    with open(file_des, "w", encoding="utf-8") as write_f:
        if not strings:
            for line in data_list:
                for word in line:
                    write_f.write(str(word) + " ")
                write_f.write("\n")
        else:
            for line in data_list:
                write_f.write(line)
                write_f.write("\n")


def save_line_as_txt(file_des, line):
    """ Saves data_line as csv in file_des """
    mode = ""
    if os.path.exists(file_des):
        mode = "a"
    else:
        mode = "w"

    with open(file_des, mode, encoding="utf-8") as write_f:
        write_f.write(line)
        write_f.write("\n")