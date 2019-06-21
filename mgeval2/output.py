def savefig(fig, fname):
    print("Saving figure to %s" % fname)
    fig.savefig(fname, bbox_inches='tight')
