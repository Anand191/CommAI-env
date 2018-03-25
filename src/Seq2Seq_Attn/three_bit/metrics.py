class Metrics(object):
    def __init__(self):
        self.err_msg = "Length Mismatch Error Between Outputs and Targets"

    def word_level(self,outputs, targets):
        if(len(outputs) != len(targets)):
            print (self.err_msg)
        word_acc = 0
        for i in range(len(outputs)):
            if(outputs[i] == targets[i]):
                word_acc += 1
        return word_acc

    def seq_level(self,outputs, targets):
        seq_acc = 0
        if (len(outputs) != len(targets)):
            print(self.err_msg)
        matches = 0
        for i in range(len(outputs)):
            if(outputs[i] == targets[i]):
                matches += 1
        if(matches == len(targets)):
            seq_acc = 1
        return  seq_acc

    def final_target(self, outputs, targets):
        acc = 0
        if (len(outputs) != len(targets)):
            print(self.err_msg)
        if(outputs[-1] == targets[-1]):
            acc = 1
        return acc
