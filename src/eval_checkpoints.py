import os
from argparse import ArgumentParser
from utility import cur_dir, read_from_file, save_line_as_txt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib import rc
rc('mathtext', default='regular')

import recurrent_nn as rnn
import recurrent_dec as rnn_dec

parser = ArgumentParser()
parser.add_argument('--cpdir', nargs='?', default='', const='')
args = parser.parse_args()

cp_path = os.path.join(cur_dir ,"rnn_checkpoints", args.cpdir)

files_enc = []
files_dec = []
train_loss = {}
train_acc = {}
dev_loss = {}
dev_bleu = {}

for file in os.listdir(cp_path):
    if file.endswith(".hdf5"):
        if file.startswith("encoder"):
            files_enc.append(file)
        if file.startswith("decoder"):
            files_dec.append(file)

cp_dir_name = str(cp_path.split("/")[-1])


for e, d in zip(sorted(files_enc), sorted(files_dec)):
    if not e[8:]==d[8:]:
        print(f"Some CP file seems to be missing! Check directory '{cp_dir_name}' for missing files!")
        exit()
    act_epoch = int(e.split("epoch")[1].split("-")[0])
    train_loss[act_epoch] = float(e.split("loss")[1].split("-acc")[0])
    train_acc[act_epoch] = float(e.split("acc")[1].split(".hdf5")[0])
    
    rnn.init_dics()
    source = read_from_file(
        os.path.join(cur_dir, "nmt_data", "multi30k.dev_subword_7000.de")
    )
    target = read_from_file(os.path.join(cur_dir, "test_data", "multi30k.dev.en"))


    enc_path = os.path.join(cp_path, e)
    res, bleu = rnn_dec.bleu_score(source, target, 100, path=enc_path)
    print("BLEU for Epoch",int(e.split("epoch")[1].split("-")[0]),"is:",bleu)
    dev_bleu[act_epoch] = bleu*100

    #print(len(target))
    #print(len(res))
    #print(res)

    #loss = rnn.categorical_loss(real, dec_output)

x = sorted(train_loss.keys())
y_train_loss = [train_loss[a] for a in x]
y_train_acc = [train_acc[a] for a in x]
#y_dev_loss = [dev_loss[a] for a in x]
y_dev_bleu = [dev_bleu[a] for a in x]
#y_dev_bleu = [12.0, 14.0, 15.9, 16.0, 18.2, 20.0, 21.0, 22.0, 27.0, 29.3]

fig = plt.figure()

ax = fig.add_subplot(111)
ax.plot(x, y_train_loss, 'x-', color='b', label = 'train loss')
ax.plot(x, y_train_acc, 'x-', color='g', label = 'train accuracy')

ax2 = ax.twinx()
ax2.plot(x, y_dev_bleu, 'x-', color='r', label = 'dev BLEU')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)

ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlabel("Epoch")
ax.set_ylabel('Loss / Accuracy')
ax2.set_ylabel('BLEU')
fig.suptitle('Model - GRU (forward) with Self Attention and Layer Norm'\
    +f'\n- Units: 500, BPE: 7000 -\nBest BLEU Dev: {max(y_dev_bleu):.2f} (Epoch {x[y_dev_bleu.index(max(y_dev_bleu))]})', fontsize=12)
plt.tight_layout(pad=2.0, rect=(0, 0, 1, 0.9))

img_path = os.path.join(cur_dir ,"plots", cp_dir_name+"_fig1.png")

plt.savefig(img_path)
#plt.show()


data_path = os.path.join(cur_dir ,"plots", cp_dir_name+".data")
save_line_as_txt(data_path, "Epoch | Train loss | Train Accuracy | Dev BLEU")
for e, l, a, b in zip(x, y_train_loss, y_train_acc, y_dev_bleu):
    save_line_as_txt(data_path, f"{e}, {l}, {a}, {b}")
