import csv


from metrics import get_word_len_avr
import os
import utility as ut
from utility import cur_dir
import math
from dictionary import Dictionary, dic_tar
from dictionary import dic_src

############## Globals ###############
# used to store directory of script file
output_filename = "output/batch"
save_batch = None

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
        """gets source windows
        size 2 * w + 1"""
        return self._source

    @property
    def label(self):
        """gets target labels"""
        return self._label

    @property
    def target(self):
        """gets target windows
        size w"""
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


def get_pred_batch(line, dev=0, window=2):
    """method creates a batch structure using only the line and its length"""
    # line is from the source file
    # create source window 2*window+1 and extend it using the alignment.
    # get firstly line deviation if not given
    if dev == 0:
        c_en, c_de = 0, 0
        dev_en = 0
        for f in os.listdir("./train_data"):
            if f.endswith(".de"):
                dev += get_word_len_avr("./train_data/" + f)
                c_de += 1
            if f.endswith(".en"):
                dev_en += get_word_len_avr("./train_data/" + f)
                c_en += 1
        # calculate deviaton
        dev = (dev_en / c_en) / (dev / c_de)

    # print(round(dev * len(line)), (dev * len(line)))
    art_tar = [0 for _ in range(round(dev * len(line)))]
    return create_batch(Batch(), line, art_tar, w=window)


def save_batch_as_int(batch):
    """writes the batch as ints inside batch.csv"""
    global output_filename
    file_des = os.path.join(cur_dir, output_filename)
    with open(file_des, "a+", newline="", encoding="utf-8") as batch_csv:
        writer = csv.writer(batch_csv)
        for (s, t, l) in zip(batch.source, batch.target, batch.label):
            writer.writerow([str(s)[1:-1], str(t)[1:-1], l])
        writer.writerow([])
    batch_csv.close()


def save_batch_as_string(batch):
    """writes the batch as strings inside batch.csv"""
    global output_filename
    file_des = os.path.join(cur_dir, output_filename)
    with open(file_des, "a+", newline="", encoding="utf-8") as batch_csv:
        writer = csv.writer(batch_csv)

        tar_keys = dic_tar.get_keys()
        src_keys = dic_src.get_keys()

        for (s, t, l) in zip(batch.source, batch.target, batch.label):
            writer.writerow(
                [
                    " ".join([src_keys[i] for i in s]),
                    " ".join([tar_keys[i] for i in t]),
                    tar_keys[l],
                ]
            )
        writer.writerow([])
    batch_csv.close()


# target and source passed as lines
def create_batch(batch, source, target, w=2):
    """
    called by create_batches,
    creates the batch and stores data in batch.csv,
    returns batch
    """
    global save_batch

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
        # if batch.size == 200:
        #     save_batch(batch)
        #     batch = Batch()

    return batch


def get_word_index(src, trg):
    """
    uses dictionaries to replace strings with the index
    """
    target, source = [], []
    dic_src.update("§$", "§$")
    dic_tar.update("§$", "§$")
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
    dic_src.store_dictionary("source_dictionary")
    dic_tar.store_dictionary("target_dictionary")
    return source, target


# as_string determines if batch is saved with int or strings
# start and end are the range of lines we create batches for
def create_batches(sor_file, tar_file, window=2, as_string=False, start=0, end=-1):
    """
    creates the batches by computing source windows, target windows
    and target label for the specified files
    """
    global save_batch
    global output_filename

    if start != 0 or end != -1:
        output_filename = "output/batch_" + str(start) + "_" + str(end)
    if as_string:
        save_batch = save_batch_as_string
        output_filename += "_string.csv"
    else:
        save_batch = save_batch_as_int
        output_filename += "_int.csv"
    # remove the batch.csv in output in case it exists
    try:
        os.remove(os.path.join(cur_dir, output_filename))
    except:
        print("No file, creating new file")

    # store source and target file as list of words
    src = ut.read_from_file(sor_file, start, end)
    trg = ut.read_from_file(tar_file, start, end)

    source, target = get_word_index(src, trg)

    batch = Batch()
    for s, t in zip(source, target):
        batch = create_batch(batch, s, t, window)
        if batch.size >= 200:
            newbatch = Batch()
            for i in range(200, batch.size):
                newbatch.append_s(batch.source[200])
                batch.source.remove(batch.source[200])
                newbatch.append_l(batch.label[200])
                batch.label.remove(batch.label[200])
                newbatch.append_t(batch.target[200])
                batch.target.remove(batch.target[200])
                newbatch.size += 1
                batch.size -= 1
            # TODO bearbeitung des batches
            batch = newbatch

    # save last batch
    if batch.size != 0:
        save_batch(batch)


def get_next_batch(batch, s, t, w=2):
    """
    Creates the next batch
    Called on the fly, see test_nn.main()
    """
    batch = create_batch(batch, s, t, w)

    # make sure batch size exceeds 200 lines
    # while batch.size < 200:
    #     batch = create_batch(batch, s, t, w)

    # remove unnecessary lines from batch to maintain 200 lines per batch
    if batch.size >= 200:
        newbatch = Batch()
        for i in range(200, batch.size):
            newbatch.append_s(batch.source[200])
            batch.source.remove(batch.source[200])
            newbatch.append_l(batch.label[200])
            batch.label.remove(batch.label[200])
            newbatch.append_t(batch.target[200])
            batch.target.remove(batch.target[200])
            newbatch.size += 1
            batch.size -= 1

        # Hold on to the excesive lines for next batch
        return batch, newbatch

    # returns None when batch contains exactly 200 lines
    return batch, None


def get_all_batches(source, target, w):
    batch = Batch()
    for s, t in zip(source, target):
        batch = create_batch(batch, s, t, w)
    return batch


def get_max_line(source, target):
    src = ut.read_from_file(source)
    tar = ut.read_from_file(target)
    len_list = map((lambda x: len(x.split())), (src + tar))
    return max(len_list)


def create_batch_rnn(source, target):
    max_line = get_max_line(source, target)
    src = ut.read_from_file(source)
    tar = ut.read_from_file(target)
    src, tar = get_word_index(src, tar)
    batch = Batch()
    for s, t in zip(src, tar):
        batch.append_s(
            list(
                reversed(
                    [dic_src.get_index("<s>")]
                    + s
                    + [dic_src.get_index("</s>")]
                    + [dic_src.get_index("</s>") for i in range(max_line - len(s) + 1)]
                )
            )
        )
        batch.append_t(
            [dic_tar.get_index("<s>")]
            + t
            + [dic_tar.get_index("</s>")]
            + [0 for i in range(max_line - len(t))]
        )
        batch.append_l(
            [dic_tar.get_index("<s>")]
            + t
            + [dic_tar.get_index("</s>")]
            + [0 for i in range(max_line - len(t))]
        )

    return batch


def main():
    # call method
    # create_batches(
    #     os.path.join(cur_dir, "data_exercise_2", "multi30k.en"),
    #     os.path.join(cur_dir, "data_exercise_2", "multi30k.de"),
    #     2,
    #     as_string=False,
    #     start=100,
    #     end=500,
    # )
    print(
        get_max_line(
            os.path.join(cur_dir, "train_data", "multi30k_subword.en"),
            os.path.join(cur_dir, "train_data", "multi30k_subword.de"),
        )
    )
    # batch = create_batch_rnn(
    #     os.path.join(cur_dir, "train_data", "multi30k.en"),
    #     os.path.join(cur_dir, "train_data", "multi30k.de"),
    # )
    # batch.toString()


if __name__ == "__main__":
    main()
