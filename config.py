from utils import is_colab, is_cluster, is_abacus
import os

EPOCHS = 50
BATCH_SIZE = 128

datasets = {
    "folksmall_melody": {
        "path": "D:\\data\\thesis_model2_dist\\folkdataset_small\\dataset.dat",
        "shape":(3711, 64, 49)
    },
    "folksmall_pianoroll": {
        "path": "D:\\data\\thesis_model2_dist\\folkdataset_small\\pianoroll.dat",
        "shape":(3715, 64, 47)
    },
    "hook100k_melody": {
        "path": "D:/data/hooktheory_dataset/00melody/hook100k_melody.dat",
        "shape": (93667, 64, 90)
    },
    "folk100k_melody": {
        "path": "D:/data/folkdataset/00melody/folk100k_melody.dat",
        "shape": (100000, 64, 58)
    },
    "folk15k_melody": {
        "path": "D:/data/folkdataset/00melody/folk15k_melody.dat",
        "shape": (15000, 64, 58)
    },
    "folk30k": {
        "path": "D:\\data\\folkdataset\\00melody\\folk30k.dat.npy",
        "shape": (30000, 64, 58)
    },
    "folk75k": {
        "path": "D:\\data\\folkdataset\\00melody\\folk75k.dat.npy",
        "shape": (75000, 64, 58)
    },
    "folk45k": {
        "path": "D:\\data\\folkdataset\\00melody\\folk45k.dat.npy",
        "shape": (45000, 64, 58)
    },
    "folk60k": {
        "path": "D:\\data\\folkdataset\\00melody\\folk60k.dat.npy",
        "shape": (60000, 64, 58)
    },
    # pianorolls
    "hook100k_pianoroll": {
        "path": "D:/data/hooktheory_dataset/00pianoroll/hook100k_pianoroll.dat",
        "shape": (95661, 64, 88)
    },
    "folk100k_pianoroll": {
        "path": "D:/data/folkdataset/00pianoroll/folk100k_pianoroll.dat",
        "shape": (100000, 64, 56)
    },
    # both
    "both_pianoroll": {
        "path": "D:\\data\\hooktheory_dataset\\BOTH_pianoroll.dat",
        "shape": (195661, 64, 88)
    }
}

if is_colab():
    datasets["hook100k_pianoroll"]['path'] = "/gdrive/My Drive/THESIS/hook100k_pianoroll.dat"

    datasets["folk100k_pianoroll"]['path'] = "/gdrive/My Drive/THESIS/folk100k_pianoroll.dat"

    datasets["folk100k_melody"]['path'] = "/gdrive/My Drive/THESIS/folk100k_melody.dat"

    datasets["hook100k_melody"]['path'] = "/gdrive/My Drive/THESIS/hook100k_melody.dat"

    datasets["folk30k"]['path'] = "/gdrive/My Drive/THESIS/folk30k.dat.npy"

    datasets["folk45k"]['path'] = "/gdrive/My Drive/THESIS/folk45k.dat.npy"

    datasets["folk60k"]['path'] = "/gdrive/My Drive/THESIS/folk60k.dat.npy"

    datasets["folk75k"]['path'] = "/gdrive/My Drive/THESIS/folk75k.dat.npy"



    

if is_cluster():
    datasets["hook100k_pianoroll"]['path'] = "/zhome/6c/8/81676/waveload2019/data/hook100k_pianoroll.dat"

    datasets["folk100k_pianoroll"]['path'] = "/zhome/6c/8/81676/waveload2019/data/folk100k_pianoroll.dat"

if is_abacus():
    datasets["hook100k_melody"]['path'] = "/scratch/kumuge/hook100k_melody.dat"
    datasets["folk100k_melody"]['path'] = "/scratch/kumuge/folk100k_melody.dat"

    datasets["hook100k_pianoroll"]['path'] = "/scratch/kumuge/hook100k_pianoroll.dat"
    datasets["folk100k_pianoroll"]['path'] = "/scratch/kumuge/folk100k_pianoroll.dat"
    datasets["both_pianoroll"]['path'] = "/scratch/kumuge/BOTH_pianoroll.dat"

    datasets["folk15k_melody"]["path"] = "/scratch/kumuge/folk15k_melody.dat"

    datasets["folk30k"]['path'] = "/scratch/kumuge/folk30k.dat.npy"

    datasets["folk45k"]['path'] = "/scratch/kumuge/folk45k.dat.npy"

    datasets["folk60k"]['path'] = "/scratch/kumuge/folk60k.dat.npy"

    datasets["folk75k"]['path'] = "/scratch/kumuge/folk75k.dat.npy"

    