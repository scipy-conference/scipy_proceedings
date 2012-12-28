import glob
fs = glob.glob("*.rst")
for f in fs:
    toks = open(f).read().split()
    ntoks = len(toks)-1
    for i in range(ntoks):
        if toks[i] == toks[i+1]:
            print toks[i], toks[i+1]
