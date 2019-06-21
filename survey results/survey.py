import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

plt.style.use('ggplot')

path = "D:\\data\\thesis_model2\\survey results\\THESIS_ Symbolic Music Survey.csv"
data = np.loadtxt(path, skiprows=1, delimiter=',', dtype=object)
data = data[:, 1:]  # drop timestamps

user_experiences, u_x_counts = np.unique(data[:-2, 0], return_counts=True)
total = np.sum(u_x_counts)

plt.figure(figsize=(6, 6))
plt.pie(
    u_x_counts, 
    labels=user_experiences, 
    shadow=True, 
    autopct=lambda p: '{:.0f}'.format(p * total / 100)
)
plt.title('Distribution of users by musical experience')
plt.savefig('user_distributions.png', bbox_inches='tight')

users = {
    l: {
        'train': {
            'r':[],'m':[]
        },
        'gen': {
            'r':[],'m':[]
        }
    }  \
        for l in user_experiences if len(l.strip()) > 0
    }
print(users)

train = {'r': [], 'm': []}

gen = {'r': [], 'm': []}

for row in data[:-2]:
    u_x = row[0]
    scores = row[1:]
    for i in range(len(scores)):
        score = int(scores[i])
        if i % 2 == 0:
            label = data[-2, i + 2]
        else:
            label = data[-2, i + 1]
        if label == '1':
            # generated sample
            if i % 2 == 0:
                # rhythm
                gen['r'].append(score)
                users[u_x]['gen']['r'].append(score)
            else:
                # melody
                gen['m'].append(score)
                users[u_x]['gen']['m'].append(score)
        elif label == '0':
            # training sample
            if i % 2 == 0:
                # rhythm
                train['r'].append(score)
                users[u_x]['train']['r'].append(score)
            else:
                # melody
                train['m'].append(score)
                users[u_x]['train']['m'].append(score)


def plot_overall(train, gen):
    g_r_mean = np.mean(gen['r'])
    g_r_std = np.std(gen['r'])

    g_m_mean = np.mean(gen['m'])
    g_m_std = np.std(gen['m'])

    t_r_mean = np.mean(train['r'])
    t_r_std = np.std(train['r'])

    t_m_mean = np.mean(train['m'])
    t_m_std = np.std(train['m'])

    names = ['train rhythm', 'gen. rhythm', 'train melody', 'gen. melody']
    x_pos = np.arange(len(names))
    means = [
        t_r_mean,
        g_r_mean,
        t_m_mean,
        g_m_mean
        ]
    stds = [
        t_r_std,
        g_r_std,
        t_m_std,
        g_m_std
    ]

    plt.figure(figsize=(6,6))
    plt.bar(
        x_pos, 
        means, 
        yerr=stds, 
        align='center', 
        alpha=0.5, 
        ecolor='black', 
        color=np.concatenate([['blue']*2, ['purple']*2])
    )
    plt.xticks(x_pos, names)
    plt.title('User study results')
    plt.savefig('user_study_results.png', bbox_inches='tight')
    table = np.vstack([
        ['%.3f' %m for m in means],
        ['%.3f' %s for s in stds]
    ])
    table = pd.DataFrame(table, columns=names).transpose()
    print(table.to_latex())

    # t tests
    # rhythm
    r_ttest = ttest_ind(gen['r'], train['r'])
    print('rhythm t-test p: %.4f' %r_ttest[1], r_ttest[1])

    # melody
    m_ttest = ttest_ind(gen['m'], train['m'])
    print('melody t-test p: %.4f' %m_ttest[1], m_ttest[1])


def plot_separate(users):
    group_1_train_r = np.concatenate([
        users['0']['train']['r'], users['1']['train']['r']
    ])
    group_1_train_m = np.concatenate([
        users['0']['train']['m'], users['1']['train']['m']
    ])

    group_1_gen_r = np.concatenate([
        users['0']['gen']['r'], users['1']['gen']['r']
    ])
    group_1_gen_m = np.concatenate([
        users['0']['gen']['m'], users['1']['gen']['m']
    ])

    group_2_train_r = np.concatenate([
        users['2']['train']['r'], 
        users['3']['train']['r'],
        users['4']['train']['r']
    ])
    group_2_train_m = np.concatenate([
        users['2']['train']['m'], 
        users['3']['train']['m'],
        users['4']['train']['m']
    ])

    group_2_gen_r = np.concatenate([
        users['2']['gen']['r'], 
        users['3']['gen']['r'],
        users['4']['gen']['r']
    ])
    group_2_gen_m = np.concatenate([
        users['2']['gen']['m'], 
        users['3']['gen']['m'],
        users['4']['gen']['m']
    ])

    names = [
        'lower train rhythm', 
        'lower gen. rhythm', 
        'lower train melody',
        'lower gen. melody', 
        'upper train rhythm', 
        'upper gen. rhythm', 
        'upper train melody',
        'upper gen. melody', 
    ]

    x_pos = np.arange(len(names))
    groups = [
        group_1_train_r,
        group_1_gen_r,
        group_1_train_m,
        group_1_gen_m,
        group_2_train_r,
        group_2_gen_r,
        group_2_train_m,
        group_2_gen_m,        
    ]
    means = [
        np.mean(g) for g in groups
    ]
    stds = [
        np.std(g) for g in groups
    ]
    
    plt.figure(figsize=(10,6))
    plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.5, ecolor='black', color=np.concatenate([['red']*2, ['orange']*2, ['blue']*2, ['purple']*2]))
    plt.xticks(x_pos, names, rotation=70)
    plt.title('User study results (aggregate per expertise levels)')
    plt.savefig('user_study_results2.png', bbox_inches='tight')


    table = np.vstack([
        ['%.3f' %m for m in means],
        ['%.3f' %s for s in stds]
    ])
    table = pd.DataFrame(table, columns=names).transpose()
    print(table.to_latex())
    
plot_overall(train, gen)

plot_separate(users)
