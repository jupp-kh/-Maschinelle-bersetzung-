import metrics
import encoder
import math
import csv
import os


############## Globals ###############
# used to store directory of script file
script_dir = os.path.dirname(__file__)

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


def create_batch(source, target, w):
    """called by create_batches, creates the batch and stores data in batch.csv,
    returns the batch"""
    batch = Batch()
    modifi_target = ["<s>" for i in range(w)] + target + ["</s>"]
    target.append("</s>")
    max_bi = alignment(len(source), len(target))(len(target))
    modif_source = (
        ["<s>" for i in range(w)]
        + source
        + ["</s>" for i in range(len(source), max_bi + w + 1)]
    )
    for i in range(len(target)):
        batch.append_l(target[i])
        batch.append_t(modifi_target[i : i + w])
        b_i = alignment(len(source), len(target))(i)
        batch.append_s(modif_source[b_i : b_i + 2 * w + 1])
    #  print(modif_source[b_i : b_i + 2 * w + 1], modifi_target[i : i + w], target[i])
    #  print(i, b_i)

    return batch


def create_batches(sor_file, tar_file, window):
    """creates the batches by computing source windows, target windows
    and target label for the specified files"""

    # remove the batch.csv in output in case it exists
    try:
        os.remove(os.path.join(script_dir, "output/batch.csv"))
    except:
        print("No file, creating new file", end="\r")

    # store source and target file as list of words
    source = " ".join(metrics.read_from_file(sor_file)).split()
    target = " ".join(metrics.read_from_file(tar_file)).split()

    # DONE insert 199 words into L and append </s>
    # TODO use integers rather than strings to reduce storage
    while len(target) > 199:
        b_i = alignment(len(source), len(target))(199)
        batch = create_batch(source[:b_i], target[:199], window)
        save_batch(batch)
        # TODO save the batch for lines 1100 to 1200
        target = target[199:]
        source = source[b_i:]

    # calculate final batch
    if len(target) != 0:
        batch = create_batch(source, target, window)
        save_batch(batch)


# used to write batches from iteration 1100 to 1200 into file
def save_batch(batch):
    file_des = os.path.join(script_dir, "output/batch.csv")
    with open(file_des, "a+", newline="", encoding="utf-8") as batch_csv:
        writer = csv.writer(batch_csv)
        for (s, t, l) in zip(batch.source, batch.target, batch.label):
            writer.writerow([" ".join(s), " ".join(t), l])
        writer.writerow([])
    batch_csv.close()


def main():
    # b = alignment(8390, 179092)(200)
    # sor = [str(i) for i in range(b)]
    # tar = [str(i) for i in range(200)]
    # source = " ".join(sor).split()
    # target = " ".join(tar).split()
    # batch = create_batch(source, target, 2)
    # os.remove("batch.csv")
    # save_batch(batch)
    create_batches(
        os.path.join(script_dir, "data_exercise_2/multi30k.en"),
        os.path.join(script_dir, "data_exercise_2/multi30k.de"),
        2,
    )


if __name__ == "__main__":
    main()
