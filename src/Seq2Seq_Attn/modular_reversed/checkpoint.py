import os
import time
import shutil
import torch
import dill

class checkpoint(object):
    MODEL_NAME = 'model.pt'
    TRAINER_STATE_NAME = 'trainer_states.pt'

    def __init__(self,model,path=None):
        self.model = model
        self.path = path

    def save(self,accuracy,best_accuracy,epoch):
        is_best = bool(accuracy > best_accuracy)
        if is_best:
            print("=> Saving a new best")
            torch.save({
                'epoch': epoch
            }, os.path.join(self.path,self.TRAINER_STATE_NAME))

            torch.save(self.model, os.path.join(self.path, self.MODEL_NAME))

        else:
            print("=> Validation Accuracy did not improve")

    @classmethod
    def load(cls,path,use_cuda):
        resume_weights = os.path.join(path,cls.MODEL_NAME)
        if os.path.isfile(resume_weights):
            print("=> loading checkpoint '{}' ...".format(resume_weights))
            if use_cuda:
                model = torch.load(resume_weights)
                start_epoch = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            else:
                # Load GPU model on CPU
                model = torch.load(resume_weights, map_location=lambda storage,loc: storage)
                start_epoch = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME),
                                         map_location=lambda storage, loc: storage)

            print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights,
                                                                             start_epoch['epoch']))
            return checkpoint(model)