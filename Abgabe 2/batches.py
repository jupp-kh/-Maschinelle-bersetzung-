import metrics
import encoder
import math
import csv
import os

class Batch:

    def __init__(self):
        """initialises batch with empty matrices"""
        self._source = []
        self._label = []
        self._target = []

    @property
    def source(self):
        return self._source

    @property
    def label(self):
        return self._label

    @property
    def target(self):
        return self._target

    def append_s(self, sor_line):
        self._source.append(sor_line)

    def append_t(self, tar_line):
        self._target.append(tar_line)

    def append_l(self, tar_lebels):
        self._label.append(tar_lebels)

    
    def toString(self):
        """print batch"""
        print("     S     |     T    | L ")
        for (s,t,l) in zip(self._source, self._target, self._label):
            print(s,t,l)



# return lambda function
# gives a basic alignment by dividing the range by domain
def alignment(sor_len, tar_len):
    """Returns alignment from target to source"""
    # for range twice the size of domain
    # map 1st element index[0] to 0*r/d = 0
    # map 2nd element index[1] to 1*r/d = 2
    # etc...
    op = lambda x: math.floor(x * (tar_len / sor_len))
    return op

def creat_batch(source , target, w):

    batch = Batch()
    max_b_i = alignment(len(target),len(source))(len(target))
    modifi_sorce = ["<s>" for i in range(w)] + source + ["</s>" for i in range(len(source), max_b_i + w +1)]
    modifi_target = ["<s>" for i in range(w)] + target + ["</s>"]
    target.append('</s>')
    print(modifi_sorce,modifi_target)
    for i in range(len(target)):
        batch.append_l(target[i])
        batch.append_t(modifi_target[i : i + w])
        b_i = alignment(len(source),len(target))(i)
        batch.append_s(modifi_sorce[b_i : b_i + 2 * w + 1 ])
        print(modifi_sorce[b_i : b_i + 2 * w + 1 ],modifi_target[i : i +  w ], target[i])
    
    return batch

            
def creat_batches(sor_file, tar_file, window):
    os.remove('batch.csv')
    source = metrics.read_from_file(sor_file)
    target = metrics.read_from_file(tar_file)

    for i, (sor, tar) in enumerate(zip(source, target)):
        if(i not in range(1100,1200)):
            continue

        sor_list = sor.split()
        tar_list = tar.split()

        while(len(tar_list) > 200):
            
            b_i = alignment(len(sor_list),len(tar_list))(200)
            batch = creat_batch(sor_list[:b_i],tar_list[:200],window)
            save_batch(batch)
            tar_list = tar_list[200:]
            sor_list = sor_list[b_i:]


        batch = creat_batch(sor_list,tar_list,window)
        save_batch(batch)

def save_batch(batch):
    with open('batch.csv', 'a+',newline = '',encoding="utf-8") as batch_csv:
        writer = csv.writer(batch_csv)
        for (s,t,l) in zip(batch.source,batch.target,batch.label):
            writer.writerow([s,t,l])
        writer.writerow([])
    batch_csv.close()



def main():
    tar = "ich habe dich gern".split()
    sor = "i like you".split()
    creat_batch(sor, tar,2)
    #creat_batches('data_exercise_2/multi30k.en','data_exercise_2/multi30k.de',2)


if __name__ == "__main__":
    main()    