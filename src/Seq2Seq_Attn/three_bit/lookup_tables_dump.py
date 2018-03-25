import tasks.micro.split_lookup_tables_1 as tasks
import itertools
import numpy as np

atomic = np.arange(1,7,dtype=int).tolist()
tables2 = [7, 8]

def all_splits(atm, tab):
    master = list(itertools.product(atm, repeat=2))
    atm_tab = list(itertools.product(atm, tab))
    tab_atm = list(itertools.product(tab, atm))
    atm_tab.extend(tab_atm)
    unseen = list(itertools.product(tables2, repeat=2))
    hybrid = list(set(atm_tab) - set(unseen))
    a1 = list(itertools.permutations(atomic,1))
    a2 = list(itertools.permutations(tables2,1))
    a1.extend(a2)
    atm.extend(tab)
    longer = list(itertools.product(atm, repeat=3))

    return a1, master,hybrid, unseen, longer

def all_object_strings(atomic, train, subset, hybrid, unseen, longer):
    train_atomic = np.zeros(len(atomic), dtype=object)
    train_composed = np.zeros((len(train), 2), dtype=object)
    test_subset = np.zeros((len(subset), 2), dtype=object)
    test_hybrid = np.zeros((len(hybrid), 2), dtype=object)
    test_unseen = np.zeros((len(unseen), 2), dtype=object)
    test_longer = np.zeros((len(longer), 2), dtype=object)

    train_atomic[:] = 'LookupTaskR3D'

    train_composed[:,0] = 'FuncLookupTaskR3D'
    train_composed[:,1] = 'FuncLookupTestTaskR3D'

    test_subset[:, 0] = 'FuncLookupTaskR3D'
    test_subset[:, 1] = 'FuncLookupTestTaskR3D'

    test_hybrid[:, 0] = 'FuncLookupTaskR3D'
    test_hybrid[:, 1] = 'FuncLookupTestTaskR3D'

    test_unseen[:, 0] = 'FuncLookupTaskR3D'
    test_unseen[:, 1] = 'FuncLookupTestTaskR3D'

    test_longer[:, 0] = 'FuncLookupTaskR3D'
    test_longer[:, 1] = 'FuncLookupTestTaskR3D'

    for i in range(len(atomic)):
        train_atomic[i] += str(atomic[i][0])

    master_data = [train, subset, hybrid, unseen, longer]
    master_objs = [train_composed, test_subset, test_hybrid, test_unseen, test_longer]

    for step,data in enumerate(master_data):
        for i in range(len(data)):
            for j in range(len(data[i])):
                if (j==len(data[i])-1):
                    str_app = str(data[i][j])
                else:
                    str_app = str(data[i][j]) + '_'
                master_objs[step][i,0] += str_app
                master_objs[step][i,1] += str_app

    return train_atomic, train_composed, test_subset, test_hybrid, test_unseen, test_longer

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

atomic_tasks, composed_tasks, hybrid_tasks, unseen_tasks, longer_tasks = all_splits(atomic,tables2)
np.random.shuffle(composed_tasks)
train_tasks, subset_tasks = composed_tasks[0:28], composed_tasks[28:]
atomic_split, train_split, subset_split, hybrid_split, unseen_split, longer_split = all_object_strings(atomic_tasks, train_tasks,
                                                                                                       subset_tasks, hybrid_tasks,
                                                                                                       unseen_tasks, longer_tasks)

# #getting all objects
train1, train2 = all_objects(atomic_split), all_objects(train_split[:,0])
heldout = all_objects(train_split[:,1])
subset1, subset2 = all_objects(subset_split[:,0]), all_objects(subset_split[:,1])
hybrid1, hybrid2 = all_objects(hybrid_split[:,0]), all_objects(hybrid_split[:,1])
unseen1, unseen2 = all_objects(unseen_split[:,0]), all_objects(unseen_split[:,1])
longer1, longer2 = all_objects(longer_split[:,0]), all_objects(longer_split[:,1])

# #getting all dicts
#train compositions
atomic_dict = gen_table_dicts(train1, atomic_tasks, 8)
train_composed = gen_table_dicts(train2, train_tasks, 6)
#heldout compositions
heldout_composed = gen_table_dicts(heldout, train_tasks, 2)
#subset compositions
test11_subset = gen_table_dicts(subset1, subset_tasks, 6)
test12_subset = gen_table_dicts(subset2, subset_tasks, 2)
#hybrid compositions
test11_hybrid = gen_table_dicts(hybrid1, hybrid_tasks, 6)
test12_hybrid = gen_table_dicts(hybrid2, hybrid_tasks, 2)
#unseen compositions
test11_unseen = gen_table_dicts(unseen1, unseen_tasks, 6)
test12_unseen = gen_table_dicts(unseen2, unseen_tasks, 2)
#longer compositions
test11_longer = gen_table_dicts(longer1, longer_tasks, 6)
test12_longer = gen_table_dicts(longer2, longer_tasks, 2)



