import tasks.micro.split_lookup_tables_1 as tasks

def atomic():
    tables = [tasks.LookupTaskR2D1(), tasks.LookupTaskR2D2(), tasks.LookupTaskR2D3(), tasks.LookupTaskR2D4()]
    atomic = {}
    j = 1
    while (j<=4):
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
