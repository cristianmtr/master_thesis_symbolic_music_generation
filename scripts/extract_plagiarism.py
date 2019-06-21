import sys
import numpy as np
import pandas as pd


args = sys.argv[1:]

files = args[:len(args)//2]
models = args[len(args)//2:]

print(models)
print(files)

table = []

for i,file in enumerate(files):
    plagiarisms = np.loadtxt(file, delimiter='\t', dtype=object)
    scores = plagiarisms[:,-1].astype(float)[::2]

    print('is -inf : ', len(scores[np.isneginf(scores)]))

    # print('inf with mean')
    indeces = np.isneginf(scores)
    indeces = np.logical_not(indeces)
    mean = np.mean(scores[indeces])
    inf_indeces = np.isneginf(scores)
    scores[np.isneginf(scores)] = mean
    print('mean = ', np.mean(scores))
    print('std = ', np.std(scores))

    # print('inf with 0')
    # scores[inf_indeces] = 0
    # print('mean = ', np.mean(scores))
    # print('std = ', np.std(scores))
    row = [models[i]]
    row.append('%.3f' %np.mean(scores))
    row.append('%.3f' %np.std(scores))
    row.append(len(scores[inf_indeces]))
    table.append(row)

cols = ['feat.']
cols.extend(['file %s' %i for i in range(len(files))])

df = pd.DataFrame(
        table,
        columns=['model', 'mean', 'SD', '-inf']
    ).set_index('model')

# print(df.to_string())
print(df.to_latex())
