import numpy as np


tables1 = np.arange(1,7,dtype=int).tolist()
tables2 = [7, 8]

class longer_splits(object):
    def __init__(self,comp_len, type, size, seed):
        self.composition_length = comp_len
        self.split = type
        self.sample = size
        self.max_size = 0
        np.random.seed(seed)

    def _max_len(self):
        if (self.split == "seen"):
            self.max_size = len(tables1)**self.composition_length
        elif (self.split == "incremental"):
            self.max_size = ((len(tables1)+len(tables2))**(self.composition_length - 1)) * len(tables1) * self.composition_length
        elif (self.split == "new"):
            self.max_size = len(tables2)**self.composition_length
        else:
            raise ValueError("invalid split type")

    def _gettuples(self):
        master = []
        if (self.split == "seen"):
            for i in range (self.composition_length):
                master.append(np.random.choice(tables1, self.sample).tolist())
            all_tup = list(zip(*master))

        elif(self.split=="incremental"):
            for i in range (self.composition_length-1):
                master.append(np.random.choice(tables1+tables2, self.sample).tolist())
            master.append(np.random.choice(tables1, self.sample).tolist())
            np.random.shuffle(master)
            all_tup = list(zip(*master))

        elif(self.split=="new"):
            for i in range (self.composition_length):
                master.append(np.random.choice(tables2, self.sample).tolist())
            all_tup = list(zip(*master))

        else:
            raise ValueError("invalid split type")

        return all_tup

    def all_composition(self):
        self._max_len()
        # var = ''
        # if(self.sample == self.max_size):
        #     var = input("The total number of splits generated might exceed memory. Continue? Y/N")
        # if (var == 'N' or var =='n'):
        #     print('Exiting')
        #     return
        if(self.sample <= self.max_size):
            split = self._gettuples()
        else:
            self.sample = self.max_size
            split = self._gettuples()
            #raise ValueError ("sample size greater than the number of possible combinations for this split type")

        return split





