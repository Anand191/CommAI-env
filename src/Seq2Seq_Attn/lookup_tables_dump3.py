import tasks.micro.split_lookup_tables_1 as tasks
import itertools
import numpy as np

atomic = np.arange(1,7,dtype=int).tolist()
tables2 = [7, 8]

def all_splits():
    master = list(itertools.product(atomic, repeat=2))
    compositions = list(itertools.permutations(atomic, 2))
    repeated = list(set(master)-set(compositions))
    unseen = list(itertools.product(tables2,repeat=2))
    unseen.extend(repeated)
    a1 = list(itertools.permutations(atomic,1))
    a2 = list(itertools.permutations(tables2,1))
    a1.extend(a2)
    atomic.extend(tables2)
    longer = list(itertools.product(atomic, repeat=3))

    return a1, compositions, unseen, longer

def all_object_strings(atomic, composed, unseen, longer):
    train_atomic = np.zeros(len(atomic), dtype=object)
    train_composed = np.zeros((len(composed), 2), dtype=object)
    test_unseen = np.zeros((len(unseen), 2), dtype=object)
    test_longer = np.zeros((len(longer), 2), dtype=object)

    train_atomic[:] = 'LookupTaskR2D'

    train_composed[:,0] = 'FuncLookupTaskR2D'
    train_composed[:,1] = 'FuncLookupTestTaskR2D'

    test_unseen[:, 0] = 'FuncLookupTaskR2D'
    test_unseen[:, 1] = 'FuncLookupTestTaskR2D'

    test_longer[:, 0] = 'FuncLookupTaskR2D'
    test_longer[:, 1] = 'FuncLookupTestTaskR2D'

    for i in range(len(atomic)):
        train_atomic[i] += str(atomic[i][0])

    master_data = [composed, unseen, longer]
    master_objs = [train_composed, test_unseen, test_longer]

    for step,data in enumerate(master_data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if (j==len(data[i])-1):
                    str_app = str(data[i][j])
                else:
                    str_app = str(data[i][j]) + '_'
                master_objs[step][i,0] += str_app
                master_objs[step][i,1] += str_app

    return train_atomic, train_composed, test_unseen, test_longer

def all_objects(names):
    tab = []
    for i in range(len(names)):
        obj = getattr(tasks, names[i])
        tab.append(obj())
    return tab

def look(lt,t,prev,break_p):
    while True:
        xx = lt.get_next_episode()

        ipt_string = xx[0].split(':')[1].split('.')[0]
        out_string = xx[1]
        curr = ipt_string

        if (curr == prev):
            continue
        else:
            t[ipt_string] = out_string

        if (len(t) == break_p):
            break
        prev = curr
    return t

def gen_table_dicts(table, comp_tup, break_p):
    cdict = {}
    i = 1
    while (i <= len(comp_tup)):
        lt_te = table[i - 1]
        prev = 'ch'
        te = look(lt_te, {}, prev, break_p)
        if (len(comp_tup[i - 1]) == 1):
            key_te = 't{}'.format(comp_tup[i-1][0])
        elif (len(comp_tup[i - 1]) == 2):
            key_te = 't{} t{}'.format(comp_tup[i - 1][1], comp_tup[i - 1][0])
        else:
            key_te = 't{} t{} t{}'.format(comp_tup[i - 1][2], comp_tup[i - 1][1], comp_tup[i - 1][0])
        if (key_te in list(cdict.keys())):
            for k, v in te.items():
                cdict[key_te][k] = v
        else:
            cdict[key_te] = te
        i += 1
    return cdict

atomic_tasks, composed_tasks, unseen_tasks, longer_tasks = all_splits()
atomic_split, composed_split, unseen_split, longer_split = all_object_strings(atomic_tasks, composed_tasks, unseen_tasks, longer_tasks)

#getting all objects
train1, train2 = all_objects(atomic_split), all_objects(composed_split[:,0])
val = all_objects(composed_split[:,1])
unseen1, unseen2 = all_objects(unseen_split[:,0]), all_objects(unseen_split[:,1])
longer1, longer2 = all_objects(longer_split[:,0]), all_objects(longer_split[:,1])

#getting all dicts
atomic_dict = gen_table_dicts(train1, atomic_tasks, 4)
train_composed = gen_table_dicts(train2, composed_tasks, 2)
val_composed = gen_table_dicts(val, composed_tasks, 2)
#unseen compositions
test11_unseen = gen_table_dicts(unseen1, unseen_tasks, 2)
test12_unseen = gen_table_dicts(unseen2, unseen_tasks, 2)
#longer compositions
test21_longer = gen_table_dicts(longer1, longer_tasks, 2)
test22_longer = gen_table_dicts(longer2, longer_tasks, 2)



