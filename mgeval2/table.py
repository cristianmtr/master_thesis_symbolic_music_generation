import os
import numpy as np

def save_table(table, dstfolder):
    table.columns = ['mean', 'SD', 'intra-set mean', 'intra-set SD', 'mean', 'SD', 'intra-set mean', 'intra-set SD', 'KLD', 'OA']

    pitch_kld = np.mean(table.loc['PC':'PCTM','KLD'].values.astype(float))
    table.loc['pitch avg.', 'KLD'] = '%.3f' %pitch_kld

    pitch_oa = np.mean(table.loc['PC':'PCTM','OA'].values.astype(float))
    table.loc['pitch avg.', 'OA'] = '%.3f' %pitch_oa

    rhythm_kld = np.mean(table.loc['NC':'NLTM','KLD'].values.astype(float))
    table.loc['rhythm avg.', 'KLD'] = '%.3f' %rhythm_kld

    rhythm_oa = np.mean(table.loc['NC':'NLTM','OA'].values.astype(float))
    table.loc['rhythm avg.', 'OA'] = '%.3f' %rhythm_oa

    table.loc['overall avg.', 'KLD'] = '%.3f' %np.mean([pitch_kld, rhythm_kld])
    table.loc['overall avg.', 'OA'] = '%.3f' %np.mean([pitch_oa, rhythm_oa])

    print(table.to_latex())
    table.to_csv(os.path.join(dstfolder, 'table.csv'))

    with open(os.path.join(dstfolder, 'table.tex'), 'w') as f:
        f.write(table.to_latex())
