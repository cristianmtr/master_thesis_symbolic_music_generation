#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import output
import os
import glob
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from mgeval import core, utils
from sklearn.model_selection import LeaveOneOut

from table import save_table
# plt.style.use('ggplot')

# ```
# 'avg_IOI',
# 'avg_pitch_shift',
# 'bar_pitch_class_histogram',
# 'bar_used_note',
# 'bar_used_pitch',
# 'chord_dependency',
# 'note_length_hist',
# 'note_length_transition_matrix',
# 'pitch_class_transition_matrix',
# 'pitch_range',
# 'total_pitch_class_histogram',
# 'total_used_note',
# 'total_used_pitch'
# ```

# ## Absolute measurement: statistic analysis
#


def main(set1, set2, set1name, set2name, dstfolder):


    num_samples = len(set1)
    set1_eval = {
        # pitch related
        'total_used_pitch': np.zeros((num_samples, 1)),
        'total_pitch_class_histogram': np.zeros((num_samples, 12)),
        'pitch_range': np.zeros((num_samples, 1)),
        'avg_pitch_shift': np.zeros((num_samples, 1)),
        'pitch_class_transition_matrix': np.zeros((num_samples, 12, 12)),
        # rhythm
        'total_used_note': np.zeros((num_samples, 1)),
        'avg_IOI': np.zeros((num_samples, 1)),
        'note_length_hist': np.zeros((num_samples, 12)),
        'note_length_transition_matrix': np.zeros((num_samples, 12, 12)),
    }

    feat_acronyms = {
        'total_used_pitch': 'PC',
        'pitch_range': 'PR',
        'avg_pitch_shift': 'PI',
        'total_pitch_class_histogram': 'PCH',
        'pitch_class_transition_matrix': 'PCTM',
        # rhythm
        'total_used_note': 'NC',
        'avg_IOI': 'IOI',
        'note_length_hist': 'NLH',
        'note_length_transition_matrix': 'NLTM'
    }

    display_names_feats = {
        'total_used_pitch': "total used pitch (PC)",
        'total_pitch_class_histogram': "pitch class histogram (PCH)",
        'pitch_range': "pitch range (PR)",
        'avg_pitch_shift': "avg. pitch interval (PI)",
        'pitch_class_transition_matrix': 'pitch class transition matrix (PCTM)',
        # rhythm
        'total_used_note': 'note count (NC)',
        'avg_IOI': 'avg. inter-onset interval (IOI)',
        'note_length_hist': 'note length histogram (NLH)',
        'note_length_transition_matrix': 'note length transition matrix (NLTM)'
    }

    metrics_list = list(set1_eval.keys())
    table = []
    for metric in ['PC', 'PR', 'PI', 'PCH', 'PCTM', 'pitch avg.', 'NC', 'IOI', 'NLH', 'NLTM', 'rhythm avg.', 'overall avg.']:
        metric_row = [metric]
        metric_row.extend(['-']*10)
        table.append(metric_row)

    table = pd.DataFrame(np.array(table,dtype=object),
        columns=[
        "feat.",
        #
        "abs_mean1", #
        "abs_sd1", #
        "intra_set_mean1", #
        "intra_set_sd1", #
        #
        "abs_mean2", #
        "abs_sd2", #
        "intra_set_mean2", #
        "intra_set_sd2", #
        #
        "inter_set_KLD",
        "inter_set_OA"
    ]).set_index('feat.')

    for i in range(0, num_samples):
        feature = core.extract_feature(set1[i])

        for metric in metrics_list:
            set1_eval[metric][i] = getattr(
                core.metrics(), metric
            )(feature)

    # repeat for second dataset
    set2_eval = {
        # pitch related
        'total_used_pitch': np.zeros((num_samples, 1)),
        'total_pitch_class_histogram': np.zeros((num_samples, 12)),
        'pitch_range': np.zeros((num_samples, 1)),
        'avg_pitch_shift': np.zeros((num_samples, 1)),
        'pitch_class_transition_matrix': np.zeros((num_samples, 12, 12)),
        # rhythm
        'total_used_note': np.zeros((num_samples, 1)),
        'avg_IOI': np.zeros((num_samples, 1)),
        'note_length_hist': np.zeros((num_samples, 12)),
        'note_length_transition_matrix': np.zeros((num_samples, 12, 12)),
    }

    for i in range(0, num_samples):
        feature = core.extract_feature(set2[i])

        for metric in metrics_list:
            set2_eval[metric][i] = getattr(
                core.metrics(), metric
            )(feature)

    # statistic analysis: absolute measurement
    absolute_measurement = ""
    for i in range(0, len(metrics_list)):
        if "transition" not in metrics_list[i] \
            and "hist" not in metrics_list[i]:
            absolute_measurement += metrics_list[i] + ':'
            absolute_measurement += "\n" + '------------------------\n'
            absolute_measurement += "\n" + set1name

            abs_mean1 = '%.3f' %np.nanmean(set1_eval[metrics_list[i]], axis=0)[0]
            abs_sd1 = '%.3f' %np.nanstd(set1_eval[metrics_list[i]], axis=0)[0]

            table.loc[feat_acronyms[metrics_list[i]], 'abs_mean1'] = abs_mean1
            table.loc[feat_acronyms[metrics_list[i]], 'abs_sd1'] = abs_sd1

            absolute_measurement += "\n" + \
                '  mean: %s' % abs_mean1
            absolute_measurement += "\n" + \
                '  std: %s' % abs_sd1

            absolute_measurement += "\n\n" + set2name

            abs_mean2 = '%.3f' %np.nanmean(set2_eval[metrics_list[i]], axis=0)[0]
            abs_sd2 = '%.3f' %np.nanstd(set2_eval[metrics_list[i]], axis=0)[0]
            
            absolute_measurement += "\n" + \
                '  mean: %s' % abs_mean2
            absolute_measurement += "\n" + \
                '  std: %s\n\n' % abs_sd2

            table.loc[feat_acronyms[metrics_list[i]], 'abs_mean2'] = abs_mean2
            table.loc[feat_acronyms[metrics_list[i]], 'abs_sd2'] = abs_sd2

    with open(os.path.join(dstfolder, '1absolute_measurement.txt'), 'w') as f:
        f.writelines(absolute_measurement)

    # ## Relative measurement: generalizes the result among features with various dimensions
    #

    # the features are sum- marized to
    # - the intra-set distances
    # - the difference of intra-set and inter-set distances.

    # exhaustive cross-validation for intra-set distances measurement

    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    set1_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    set2_intra = np.zeros((num_samples, len(metrics_list), num_samples-1))
    for i in range(len(metrics_list)):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            distances = utils.c_dist(
                set1_eval[metrics_list[i]][test_index],
                set1_eval[metrics_list[i]][train_index]
                )
            distances_mean = np.nanmean(distances)
            distances[np.where(np.isnan(distances))] = distances_mean
            set1_intra[test_index[0]][i] = distances
            del distances

            distances = utils.c_dist(
                set2_eval[metrics_list[i]][test_index], 
                set2_eval[metrics_list[i]][train_index]
                )
            distances_mean = np.nanmean(distances)
            distances[np.where(np.isnan(distances))] = distances_mean
            set2_intra[test_index[0]][i] = distances

    # exhaustive cross-validation for inter-set distances measurement
    loo = LeaveOneOut()
    loo.get_n_splits(np.arange(num_samples))
    sets_inter = np.zeros((num_samples, len(metrics_list), num_samples))

    for i in range(len(metrics_list)):
        for train_index, test_index in loo.split(np.arange(num_samples)):
            distances = utils.c_dist(
                set1_eval[metrics_list[i]][test_index],
                set2_eval[metrics_list[i]]
                )
            distances_mean = np.nanmean(distances)
            distances[np.where(np.isnan(distances))] = distances_mean
            sets_inter[test_index[0]][i] = distances

    # visualization of intra-set and inter-set distances
    plot_set1_intra = np.transpose(
        set1_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_set2_intra = np.transpose(
        set2_intra, (1, 0, 2)).reshape(len(metrics_list), -1)
    plot_sets_inter = np.transpose(
        sets_inter, (1, 0, 2)).reshape(len(metrics_list), -1)
    for i in range(0, len(metrics_list)):
        for s in [
            plot_set1_intra[i],
            plot_set2_intra[i],
            plot_sets_inter[i]
        ]:
            s[np.isnan(s)] = np.nanmean(s)


        sns.kdeplot(plot_set1_intra[i], label='intra %s' % set1name)
        sns.kdeplot(plot_sets_inter[i], label='inter')
        sns.kdeplot(plot_set2_intra[i], label='intra %s' % set2name)

        intra_set_mean1 = '%.3f' %np.nanmean(plot_set1_intra[i], axis=0)
        intra_set_sd1 = '%.3f' %np.nanstd(plot_set1_intra[i], axis=0)

        table.loc[feat_acronyms[metrics_list[i]], 'intra_set_mean1'] = intra_set_mean1
        table.loc[feat_acronyms[metrics_list[i]], 'intra_set_sd1'] = intra_set_sd1

        intra_set_mean2 = '%.3f' %np.nanmean(plot_set2_intra[i], axis=0)
        intra_set_sd2 = '%.3f' %np.nanstd(plot_set2_intra[i], axis=0)

        table.loc[feat_acronyms[metrics_list[i]], 'intra_set_mean2'] = intra_set_mean2
        table.loc[feat_acronyms[metrics_list[i]], 'intra_set_sd2'] = intra_set_sd2

        kl = '%.3f' %utils.kl_dist(
            plot_set1_intra[i],
            plot_set2_intra[i]
        )
        oa = '%.3f' %utils.overlap_area(
            plot_set1_intra[i],
            plot_set2_intra[i]
        )
        table.loc[feat_acronyms[metrics_list[i]], 'inter_set_KLD'] = kl
        table.loc[feat_acronyms[metrics_list[i]], 'inter_set_OA'] = oa

        plt.title(display_names_feats[metrics_list[i]])
        plt.xlabel('Euclidean distance')
        plt.xlabel('Density')
        output.savefig(plt.gcf(), os.path.join(
            dstfolder, '3' + metrics_list[i] + '.png'))
        plt.clf()

    # the difference of intra-set and inter-set distances.
    distance_text = ''
    for i in range(0, len(metrics_list)):
        print(metrics_list[i])
        distance_text += metrics_list[i] + ':\n'
        distance_text += '------------------------\n' 
        distance_text += "\n" + set1name

        kl = '%.3f' %utils.kl_dist(
            plot_set1_intra[i],
            plot_sets_inter[i]
        )
        oa = '%.3f' %utils.overlap_area(
            plot_set1_intra[i],
            plot_sets_inter[i]
        )

        # table.loc[feat_acronyms[metrics_list[i]], 'inter_set_KLD'] = kl
        # table.loc[feat_acronyms[metrics_list[i]], 'inter_set_OA'] = oa

        distance_text += "\n" + '  Kullback-Leibler divergence: %s' % kl
        distance_text += "\n" + '  Overlap area: %s' % oa

        distance_text += "\n" + set2name

        plot_set2_intra_i_mean = np.nanmean(plot_set2_intra[i])
        plot_set2_intra[i][np.where(np.isnan(plot_set2_intra[i]))] = plot_set2_intra_i_mean

        kl = '%.3f' %utils.kl_dist(
                    plot_set2_intra[i], 
                    plot_sets_inter[i]
                )
        oa = '%.3f' %utils.overlap_area(
            plot_set2_intra[i], 
            plot_sets_inter[i]
        )

        distance_text += "\n" + '  Kullback-Leibler divergence: %s' % kl
        distance_text += "\n" + '  Overlap area: %s\n\n' % oa

    with open(os.path.join(dstfolder, '4distance_text.txt'), 'w') as f:
        f.writelines(distance_text)

    # save table
    save_table(table, dstfolder)

    # pitch tm
    mpl.rc('font',family = 'sans-serif',  size=20)

    note_names = [
        "C",
        'Db',
        "D",
        "Eb",
        "E",
        "F",
        "Gb",
        "G",
        "Ab",
        "A",
        "Bb",
        "B"
    ]

    plt.clf()
    plt.figure(figsize=(12,12))
    sns.heatmap(np.mean(set1_eval['pitch_class_transition_matrix'],axis=0), cmap='Blues')
    sns.set(font_scale=1.4)
    plt.xticks([i+.5 for i in range(len(note_names))], note_names)
    plt.yticks([i+.5 for i in range(len(note_names))], note_names)
    # plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title("Pitch transition matrix for %s samples" %set1name)
    plt.gcf().savefig(os.path.join(dstfolder, '5pitch_tm_%s.png' %set1name), bbox_inches='tight')

    plt.clf()
    plt.figure(figsize=(12,12))
    sns.heatmap(np.mean(set2_eval['pitch_class_transition_matrix'],axis=0), cmap='Blues')
    sns.set(font_scale=1.4)
    plt.xticks([i+.5 for i in range(len(note_names))], note_names)
    plt.yticks([i+.5 for i in range(len(note_names))], note_names)
    # plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title("Pitch transition matrix for %s samples" %set2name)
    plt.gcf().savefig(os.path.join(dstfolder, '5pitch_tm_%s.png' %set2name), bbox_inches='tight')
    # nl tm

    note_lens = [
        "$W$",
        "$H$",
        "$Q$",
        "$E$",
        "$S$",
        "$H .$",
        "$Q .$",
        "$E .$",
        "$S .$",
        "$H t$ ",
        "$Q t$",
        "$E t$",
        
    ]

    plt.clf()
    plt.figure(figsize=(12,12))
    sns.heatmap(np.mean(set1_eval['note_length_transition_matrix'],axis=0), cmap="Reds", vmax=7)
    sns.set(font_scale=1.4)
    plt.xticks([i+.5 for i in range(len(note_lens))], note_lens)
    plt.yticks([i+.5 for i in range(len(note_lens))], note_lens)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title("Note length transition matrix for %s samples" %set1name)
    plt.gcf().savefig(os.path.join(dstfolder, '5notelength_tm_%s.png' %set1name), bbox_inches='tight')

    plt.clf()
    plt.figure(figsize=(12,12))
    sns.heatmap(np.mean(set2_eval['note_length_transition_matrix'],axis=0), cmap="Reds", vmax=7)
    sns.set(font_scale=1.4)
    plt.xticks([i+.5 for i in range(len(note_lens))], note_lens)
    plt.yticks([i+.5 for i in range(len(note_lens))], note_lens)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.title("Note length transition matrix for %s samples" %set2name)
    plt.gcf().savefig(os.path.join(dstfolder, '5notelength_tm_%s.png' %set2name), bbox_inches='tight')
    plt.clf()

    ## 

    ## total_pitch_class_histogram


    ## note_length_hist

    np.save(os.path.join(dstfolder, 'set1.npy'), set1_eval)
    np.save(os.path.join(dstfolder, 'set2.npy'), set2_eval)


    print('folder : %s' %os.path.abspath(dstfolder))
    return


# Assign dataset path
if __name__ == "__main__":
    globstr1 = 'd:/thesis_code/model_fulldata1/samples/sessiontune45840_transposed/temp*.mid'
    set1 = glob.glob(
        globstr1)
    set1name = 'generated'
    print 'we have %s samples' % len(set1)
    num_samples = len(set1)

    globstr2 = 'd:/data/folkdataset/4_transposed_split_4bars/*.mid'
    set2 = glob.glob(globstr2)
    random.shuffle(set2)
    set2 = set2[:num_samples]

    set2name = 'training'

    dstfolder = 'comparison1'
    if not os.path.exists(dstfolder):
        os.mkdir(dstfolder)

    with open(os.path.join(dstfolder, '0info.txt'), 'w') as f:
        f.writelines("Comparison between %s and %s" %(globstr1, globstr2))

    main(set2, set1, set2name, set1name, dstfolder) # order matters in KL
