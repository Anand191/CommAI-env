import tasks.micro.split_lookup_tables_1 as tasks

def atomic():
    tables = [tasks.LookupTaskR2D1(), tasks.LookupTaskR2D2(), tasks.LookupTaskR2D3(), tasks.LookupTaskR2D4()]
    #tasks.LookupTaskR2D5(), tasks.LookupTaskR2D6()
    atomic = {}
    j = 1
    while (j<=len(tables)):
        lt = tables[j-1]
        i= 0
        t = {}
        prev = 'ch'
        while True:
            xx = lt.get_next_episode()
            ipt_string = xx[0].split(':')[1].split('.')[0]
            out_string = xx[1]
            curr = ipt_string
            if (curr == prev):
                continue
            else:
                t[ipt_string] = out_string

            if(len(t)==4):
                break
            prev = curr
        key = 't{}'.format(j)
        atomic[key] = t
        j +=1
    return atomic

def composed():
    train_tables = [tasks.FuncLookupTaskR2D1_2(), tasks.FuncLookupTaskR2D1_3(), tasks.FuncLookupTaskR2D1_4(),
                    tasks.FuncLookupTaskR2D2_1(), tasks.FuncLookupTaskR2D2_3(),tasks.FuncLookupTaskR2D2_4(),
                    tasks.FuncLookupTaskR2D3_1(), tasks.FuncLookupTaskR2D3_2(), tasks.FuncLookupTaskR2D3_4(),
                    tasks.FuncLookupTaskR2D4_1(), tasks.FuncLookupTaskR2D4_2(), tasks.FuncLookupTaskR2D4_3()]

    test_tables = [tasks.FuncLookupTestTaskR2D1_2(), tasks.FuncLookupTestTaskR2D1_3(), tasks.FuncLookupTestTaskR2D1_4(),
                    tasks.FuncLookupTestTaskR2D2_1(), tasks.FuncLookupTestTaskR2D2_3(), tasks.FuncLookupTestTaskR2D2_4(),
                    tasks.FuncLookupTestTaskR2D3_1(), tasks.FuncLookupTestTaskR2D3_2(), tasks.FuncLookupTestTaskR2D3_4(),
                    tasks.FuncLookupTestTaskR2D4_1(), tasks.FuncLookupTestTaskR2D4_2(), tasks.FuncLookupTestTaskR2D4_3(),
                   tasks.FuncLookupTestTaskR2D4_4(),tasks.FuncLookupTestTaskR2D3_3(),tasks.FuncLookupTestTaskR2D2_2(),
                   tasks.FuncLookupTestTaskR2D1_1()]
                # , tasks.FuncLookupTestTaskR2D5_5(), tasks.FuncLookupTestTaskR2D5_6(),tasks.FuncLookupTestTaskR2D6_6(), tasks.FuncLookupTestTaskR2D6_5()

    comp_seq = [(1,2), (1,3), (1,4), (2,1), (2,3), (2,4), (3,1), (3,2), (3,4), (4,1), (4,2), (4,3)]
    comp_seq_t = [(1, 2), (1, 3), (1, 4), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 4), (4, 1), (4, 2), (4, 3),
                  (4, 4),(3, 3),(2, 2),(1, 1)] #,(5, 5), (5, 6), (6, 6), (6, 5)

    composed_train, composed_test = {}, {}

    def look(lt,t,prev):
        while True:
            xx = lt.get_next_episode()

            ipt_string = xx[0].split(':')[1].split('.')[0]
            out_string = xx[1]
            curr = ipt_string

            if (curr == prev):
                continue
            else:
                t[ipt_string] = out_string

            if (len(t) == 2):
                break
            prev = curr
        return t

    j = 1
    while (j <= 12):
        lt_tr = train_tables[j - 1]
        prev = 'ch'
        tr = look(lt_tr,{},prev)
        key_tr = 't{} t{}'.format(comp_seq[j-1][1], comp_seq[j-1][0])
        composed_train[key_tr] = tr
        j += 1

    i = 1
    while(i <= len(comp_seq_t)):
        lt_te = test_tables[i - 1]
        prev = 'ch'
        te = look(lt_te, {}, prev)
        if(len(comp_seq_t[i-1])==2):
            key_te = 't{} t{}'.format(comp_seq_t[i - 1][1], comp_seq_t[i - 1][0])
        else:
            key_te = 't{} t{} t{}'.format(comp_seq_t[i - 1][2], comp_seq_t[i - 1][1], comp_seq_t[i - 1][0] )
        composed_test[key_te] = te
        i += 1


    return (composed_train,composed_test)

com = composed()
