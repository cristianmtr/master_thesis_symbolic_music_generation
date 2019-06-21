import output
import os
import glob
import numpy as np
import random
from sklearn.model_selection import LeaveOneOut

from main import main

num_samples = 100

globstr1 = 'D:\\data\\folkdataset\\4_mono\\*.mid'
set1 = glob.glob(
    globstr1)
random.shuffle(set1)
set1 = set1[:num_samples]
set1name = 'folk'
print 'we have %s samples' % len(set1)
num_samples = len(set1)

globstr2 = 'D:\\data\\hooktheory_dataset\\4_mono\\*.mid'
set2 = glob.glob(globstr2)
random.shuffle(set2)
set2 = set2[:num_samples]

set2name = 'hook theory'

dstfolder = 'comparison_folk_hook'
if not os.path.exists(dstfolder):
    os.mkdir(dstfolder)

with open(os.path.join(dstfolder, '0info.txt'), 'w') as f:
    f.writelines("Comparison between %s and %s" %(globstr1, globstr2))

main(set2, set1, set2name, set1name, dstfolder) # order matters in KL
