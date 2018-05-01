import pandas as pd
import numpy as np

df = pd.read_csv('./data2/composed_train.csv', delimiter='\t')
df[['ipt','tab1', 'tab2']] = df['ipt'].str.split(' ', expand=True)
df[['copy','interim', 'opt']] = df['copy'].str.split(' ',expand=True)
df['compositions'] = df[['tab1', 'tab2']].apply(lambda x: ' '.join(x), axis=1)

arr = np.zeros((21,8))
Y= df['compositions'].unique().tolist()
Ybucket = {}
for y in Y:
    Ybucket[y] = 0
sudoku = pd.DataFrame(arr, columns=df['opt'].unique().tolist())
count = 0
total = 168
for j in range(0, sudoku.shape[0]):
    for opt in sudoku.columns:
        compositions = [x for x in list(Ybucket.keys()) if Ybucket[x] < 6]
        print(compositions)
        while True:
            c = np.random.choice(compositions, 1)
            if (c[0] not in sudoku[opt].iloc[:].values):
                break
            else:
                continue
        count += 1
        left = total-count
        print("{} found -- {} left".format(count, left))
        sudoku[opt].iloc[j] = c[0]
        Ybucket[c[0]] += 1
print(sudoku)

# arr = np.zeros((6,28), dtype=object)
# Y = df['opt'].unique().tolist()
# Ybucket = {}
# for y in Y:
#     Ybucket[y] = 0
# sudoku = pd.DataFrame(arr, columns=df['compositions'].unique().tolist())
# count = 0
# total = 168
# for j in range(sudoku.shape[0]):
#     for com in sudoku.columns:
#         outputs = [x for x in list(Ybucket.keys()) if Ybucket[x] < 8]
#         print(outputs)
#         while True:
#             op = np.random.choice(outputs, 1)
#             if(op[0] not in sudoku[com].iloc[:].values):
#                 break
#             else:
#                 continue
#         count +=1
#         left = total - count
#         print("{} found -- {} left".format(count, left))
#         sudoku[com].iloc[j] = op[0]
#         Ybucket[op[0]] += 1
#sudoku.to_csv('./data2/balanced.csv', sep='\t', index=False)
