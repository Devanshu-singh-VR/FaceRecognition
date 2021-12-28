import pandas as pd

p = pd.read_csv('train.csv')

path1 = p.iloc[:, 0]
path2 = p.iloc[:, 1]
label = p.iloc[:, 2]

mat = []
for i in range(1, 12000):
    if label[i] == 1:
        gul = set()
        for j in range(i, 40+i):
            try:
                if path2[j].split('/')[1] != path1[i].split('/')[1] and path2[j] not in gul:
                    mat.append([path1[i], path2[i], path2[j]])
                    gul.add(path2[j])
            except:
                continue

df = pd.DataFrame(mat, columns=['face1', 'face2', 'face3'])
df.to_csv('train_triplet.csv', index=False)