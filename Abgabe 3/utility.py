import sys, time, threading
import os
import csv

cur_dir = os.path.dirname(__file__)

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
    try:
        os.remove(file_des)
    except:
        print("No file, creating new file")
    
    with open(file_des, "x", encoding="utf-8") as write_f:
        write_f.write(data)