import torch
import pandas as pd

file_path = []

for i in range(1620, 1680):
    print(i)
    recurrent = set()
    for j in range(0, 4):
        a = 'images/' + str(i) + '/' + str(j) + '.jpg'
        try:
            open(a)
            for k in range(i, i+4):
                if k == i:
                    p = 14
                else:
                    p = 1
                for l in range(0, p):
                    try:
                        if k == i and j == l:
                            continue
                        elif k == i and j != l:
                            if k in recurrent:
                                continue
                            else:
                                open('images/' + str(k) + '/' + str(l) + '.jpg')
                                file_path.append([a, 'images/' + str(k) + '/' + str(l) + '.jpg', 1])
                            if l == 3:
                                recurrent.add(k)
                        else:
                            open('images/' + str(k) + '/' + str(l) + '.jpg')
                            file_path.append([a, 'images/' + str(k) + '/' + str(l) + '.jpg', 0])
                    except:
                        continue
        except:
            continue

dataframe = pd.DataFrame(file_path, columns=['face1', 'face2', 'label'])

dataframe.to_csv('test.csv', index=False)