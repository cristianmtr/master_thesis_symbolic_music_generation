import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def plot_att():
    path_to_attention = "D:\\data\\thesis_model2\\model_folk100k_melody_2lstm32_attention\\samples\\blood_transposed\\atts.npy"

    att = np.load(path_to_attention)
    print(att.shape)
    heatmap = []
    for s in range(att.shape[1]):
        step = np.mean(att[:,s],axis=0)
        heatmap.append(step)

    # heatmap = np.array(heatmap)[:63]
    plt.figure(figsize=(12,12))
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax = sns.heatmap(heatmap,cmap="Reds",vmax=0.4,cbar=False)
    sns.set(font_scale=1.4)
    plt.ylabel('Time step to generate',fontsize=18)
    plt.xlabel('Attention weights across inputs',fontsize=18)
    plt.title('Attention across time step generation')
    plt.gcf().savefig('attention.png',bbox_inches='tight')


def plot_softmax():
    path_to_softmax = "D:\\data\\thesis_model2\\model_folk100k_melody_2lstm32_noattention\\samples\\blood_transposed\\softmax.npy"

    soft = np.load(path_to_softmax)
    print(soft.shape)
    plt.figure(figsize=(12,12))
    plt.tick_params(axis='both', which='major', labelsize=16)
    sns.heatmap(np.mean(soft,axis=0),cmap='Reds',cbar=False)
    sns.set(font_scale=1.4)
    plt.ylabel('Time step in generated sequence',fontsize=18)
    plt.xlabel('Softmax across note range in vocabulary',fontsize=18)
    plt.title('Softmax heatmap of base model (no attention)')
    plt.gcf().savefig('softmax_noattention.png',bbox_inches='tight')



if __name__ == "__main__":
    plot_att()
    plot_softmax()