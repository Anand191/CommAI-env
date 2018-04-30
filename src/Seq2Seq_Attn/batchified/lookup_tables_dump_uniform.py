import tasks.micro.split_lookup_tables_1 as tasks
import itertools
import numpy as np


class lookup_tables(object):
    def __init__(self,str0, str1, str2, atomic, tables2):
        self.obj0 = str0
        self.obj1 = str1
        self.obj2 = str2
        self.atomic = atomic
        self.tables2 = tables2

    def all_splits(self, atm, tab):
        master = list(itertools.product(atm, repeat=2))
        atm_tab = list(itertools.product(atm, tab))
        tab_atm = list(itertools.product(tab, atm))
        atm_tab.extend(tab_atm)
        unseen = list(itertools.product(self.tables2, repeat=2))
        hybrid = list(set(atm_tab) - set(unseen))
        a1 = list(itertools.permutations(self.atomic,1))
        a2 = list(itertools.permutations(self.tables2,1))
        a1.extend(a2)
        atm.extend(tab)

        return a1, master,hybrid, unseen

    def all_object_strings(self, atomic, train, subset, hybrid, unseen):
        train_atomic = np.zeros(len(atomic), dtype=object)
        train_composed = np.zeros((len(train), 2), dtype=object)
        test_subset = np.zeros((len(subset), 2), dtype=object)
        test_hybrid = np.zeros((len(hybrid), 2), dtype=object)
        test_unseen = np.zeros((len(unseen), 2), dtype=object)

        train_atomic[:] = self.obj0

        train_composed[:,0] = self.obj1
        train_composed[:,1] = self.obj2

        test_subset[:, 0] = self.obj1
        test_subset[:, 1] = self.obj2

        test_hybrid[:, 0] = self.obj1
        test_hybrid[:, 1] = self.obj2

        test_unseen[:, 0] = self.obj1
        test_unseen[:, 1] = self.obj2

        for i in range(len(atomic)):
            train_atomic[i] += str(atomic[i][0])

        master_data = [train, subset, hybrid, unseen]
        master_objs = [train_composed, test_subset, test_hybrid, test_unseen]

        for step,data in enumerate(master_data):
            for i in range(len(data)):
                for j in range(len(data[i])):
                    if (j==len(data[i])-1):
                        str_app = str(data[i][j])
                    else:
                        str_app = str(data[i][j]) + '_'
                    master_objs[step][i,0] += str_app
                    master_objs[step][i,1] += str_app

        return train_atomic, train_composed, test_subset, test_hybrid, test_unseen

    def all_objects(self, names):
        tab = []
        for i in range(len(names)):
            obj = getattr(tasks, names[i])
            tab.append(obj())
        return tab

    def look(self, lt,t,prev,break_p):
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

    def gen_table_dicts(self, table, comp_tup, break_p):
        cdict = {}
        i = 1
        while (i <= len(comp_tup)):
            lt_te = table[i - 1]
            prev = 'ch'
            te = self.look(lt_te, {}, prev, break_p)
            if (len(comp_tup[i - 1]) == 1):
                key_te = 't{}'.format(comp_tup[i-1][0])
            # elif (len(comp_tup[i - 1]) == 2):
            #     key_te = 't{} t{}'.format(comp_tup[i - 1][1], comp_tup[i - 1][0])
            else:
                key_te = 't{}'.format(comp_tup[i-1][-1])
                for j in range(len(comp_tup[i-1])-2, -1,-1):
                    key_te += ' t{}'.format(comp_tup[i-1][j])
            if (key_te in list(cdict.keys())):
                for k, v in te.items():
                    cdict[key_te][k] = v
            else:
                cdict[key_te] = te
            i += 1
        return cdict

    def gen_all_data(self):
        atomic_tasks, composed_tasks, hybrid_tasks, unseen_tasks = self.all_splits(self.atomic, self.tables2)
        np.random.shuffle(composed_tasks)
        train_tasks, subset_tasks = composed_tasks[0:28], composed_tasks[28:]
        atomic_split, train_split, subset_split, hybrid_split, unseen_split = self.all_object_strings(atomic_tasks, train_tasks,
                                                                                                               subset_tasks, hybrid_tasks,
                                                                                                               unseen_tasks)

        # #getting all objects
        train1, train2 = self.all_objects(atomic_split), self.all_objects(train_split[:,0])
        train3 = self.all_objects(train_split[:,1])
        subset1, subset2 = self.all_objects(subset_split[:,0]), self.all_objects(subset_split[:,1])
        hybrid1, hybrid2 = self.all_objects(hybrid_split[:,0]), self.all_objects(hybrid_split[:,1])
        unseen1, unseen2 = self.all_objects(unseen_split[:,0]), self.all_objects(unseen_split[:,1])

        # #getting all dicts
        #train compositions
        atomic_dict = self.gen_table_dicts(train1, atomic_tasks, 8)
        train_composed = self.gen_table_dicts(train2, train_tasks, 6)

        #heldout compositions
        train_composed2 = self.gen_table_dicts(train3, train_tasks, 2)

        #subset compositions
        test11_subset = self.gen_table_dicts(subset1, subset_tasks, 6)
        test12_subset = self.gen_table_dicts(subset2, subset_tasks, 2)

        #hybrid compositions
        test11_hybrid = self.gen_table_dicts(hybrid1, hybrid_tasks, 6)
        test12_hybrid = self.gen_table_dicts(hybrid2, hybrid_tasks, 2)

        #unseen compositions
        test11_unseen = self.gen_table_dicts(unseen1, unseen_tasks, 6)
        test12_unseen = self.gen_table_dicts(unseen2, unseen_tasks, 2)

        return (atomic_dict, train_composed, train_composed2, test11_subset, test12_subset, test11_hybrid,
                test12_hybrid, test11_unseen, test12_unseen)



