import csv
import os
import utility as ut
from utility import cur_dir
import dictionary
import math


############## Globals ###############
# used to store directory of script file
dic_tar = dictionary.Dictionary()
dic_src = dictionary.Dictionary()

# to create our batches we build the three multidimensional arrays
# S : B x (w * 2 + 1)
# T : B x w
# L : B
class Batch:
    def __init__(self):
        """initialises batch with empty matrices"""
        self._source = []
        self._label = []
        self._target = []
        self._size = 0

    @property
    def source(self):
        """gets source windows"""
        return self._source

    @property
    def label(self):
        """gets target labels"""
        return self._label

    @property
    def target(self):
        """gets target windows"""
        return self._target

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, var):
        self._size = var

    def append_s(self, sor_line):
        self._source.append(sor_line)

    def append_t(self, tar_line):
        self._target.append(tar_line)

    def append_l(self, tar_lebels):
        self._label.append(tar_lebels)

    def toString(self):
        """prints batch"""
        print("     S     |     T    | L ")
        for (s, t, l) in zip(self._source, self._target, self._label):
            print(s, t, l)


# return lambda function
# gives a basic alignment by dividing the range by domain
def alignment(sor_len, tar_len):
    """Returns alignment from target to source"""
    # for range twice the size of domain
    # map 1st element index[0] to 0*r/d = 0
    # map 2nd element index[1] to 1*r/d = 2
    # etc...
    op = lambda x: math.floor(x * (sor_len / tar_len))
    return op


# target and source passed as lines
def create_batch(batch, source, target, w):
    """called by create_batches,
    creates the batch and stores data in batch.csv,
    returns batch"""
    modifi_target = (
        [dic_tar.get_index("<s>") for i in range(w)]
        + target
        + [dic_tar.get_index("</s>")]
    )
    target.append(dic_tar.get_index("</s>"))
    max_bi = alignment(len(source), len(target))(len(target))
    modif_source = (
        [dic_src.get_index("<s>") for i in range(w)]
        + source
        + [dic_src.get_index("</s>") for i in range(len(source), max_bi + w + 1)]
    )
    for i in range(len(target)):
        batch.size += 1
        batch.append_l(target[i])
        batch.append_t(modifi_target[i : i + w])
        b_i = alignment(len(source), len(target))(i)
        batch.append_s(modif_source[b_i : b_i + 2 * w + 1])
        if batch.size == 200:
            save_batch(batch)
            batch = Batch()

    return batch


#
def get_word_index(src, trg):
    target, source = [], []
    dic_src.update("<s>", "</s>")
    dic_tar.update("<s>", "</s>")

    # creating index for each word
    # mapping lines to list of words
    for trg_l, src_l in zip(trg, src):
        tmp_t, tmp_s = [], []
        for trg_w in trg_l.split():
            dic_tar.update(trg_w)
            tmp_t.append(dic_tar.get_index(trg_w))
        for src_w in src_l.split():
            dic_src.update(src_w)
            tmp_s.append(dic_src.get_index(src_w))
        target.append(tmp_t)
        source.append(tmp_s)

    return source, target


def create_batches(sor_file, tar_file, window, start=0, end=-1):
    """creates the batches by computing source windows, target windows
    and target label for the specified files"""

    # remove the batch.csv in output in case it exists
    try:
        os.remove(os.path.join(cur_dir, "output/batch.csv"))
    except:
        print("No file, creating new file")

    # store source and target file as list of words
    src = ut.read_from_file(sor_file, start, end)
    trg = ut.read_from_file(tar_file, start, end)

    source, target = get_word_index(src, trg)

    #
    size = 0
    batch = Batch()
    for s, t in zip(source, target):
        batch = create_batch(batch, s, t, 2)

    # save last batch
    if batch.size != 0:
        save_batch(batch)


# used to write batches from iteration 1100 to 1200 into file
def save_batch(batch):
    file_des = os.path.join(cur_dir, "output/batch.csv")
    with open(file_des, "a+", newline="", encoding="utf-8") as batch_csv:
        writer = csv.writer(batch_csv)
        for (s, t, l) in zip(batch.source, batch.target, batch.label):
            writer.writerow([str(s)[1:-1], str(t)[1:-1], l])
        writer.writerow([])
    batch_csv.close()


def export(will=False):
    des = os.path.join(cur_dir, "output/string.csv")
    origin = ut.read_from_file(os.path.join(cur_dir, "output/batch.csv"))
    with open(des, "w", newline="", encoding="utf-8") as string_csv:
        for line in origin:
            for word in line:
                line.replace(word, di)


def main():
    create_batches(
        os.path.join(cur_dir, "data_exercise_2/multi30k.en"),
        os.path.join(cur_dir, "data_exercise_2/multi30k.de"),
        2,
    )

    export(arg[1])


if __name__ == "__main__":
    main()
