import os
import time
import shutil
import torch
import dill

class checkpoint(object):
    """
      The Checkpoint class manages the saving and loading of a model during training. It allows training to be suspended
      and resumed at a later time (e.g. when running on a cluster using sequential jobs).

      To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
      to write parameters to disk.

      Args:
          model: encoder/decoder being trained
          optimizer (Optimizer): stores the state of the optimizer
          epoch (int): current epoch (an epoch is a loop through the full training data)
          step (int): number of examples seen within the current epoch
          input_vocab (Vocabulary): vocabulary for the input language
          output_vocab (Vocabulary): vocabulary for the output language

      Attributes:
          TRAINER_STATE_NAME (str): name of the file storing trainer states
          MODEL_NAME (str): name of the file storing model
          INPUT_VOCAB_FILE (str): name of the input vocab file
          OUTPUT_VOCAB_FILE (str): name of the output vocab file
      """
    MODEL_NAME = 'model.pt'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    INPUT_VOCAB_FILE = 'input_vocab.pt'
    OUTPUT_VOCAB_FILE = 'output_vocab.pt'

    def __init__(self,model, optimizer, input_vocab, output_vocab, path=None):
        self.model = model
        self.optimizer = optimizer
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved.")
        return self._path

    def save(self,loss,best_loss, epoch):
        path = self._path
        is_best = bool(loss < best_loss)
        if is_best:
            print("=> Saving a new best")
            torch.save({
                'epoch': epoch,
                'optimizer': self.optimizer
            }, os.path.join(self.path,self.TRAINER_STATE_NAME))

            torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        else:
            print("=> Validation Loss did not decrease")

        with open(os.path.join(path, self.INPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.input_vocab, fout)
        with open(os.path.join(path, self.OUTPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.output_vocab, fout)

    @classmethod
    def load(cls,path,use_cuda):
        resume_weights = os.path.join(path,cls.MODEL_NAME)
        print("=> loading checkpoint '{}' ...".format(resume_weights))
        if use_cuda:
            model = torch.load(resume_weights)
            start_epoch = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
        else:
            # Load GPU model on CPU
            model = torch.load(resume_weights, map_location=lambda storage,loc: storage)
            start_epoch = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME),
                                     map_location=lambda storage, loc: storage)

        #model.flatten_parameters()  # make RNN parameters contiguous
        with open(os.path.join(path, cls.INPUT_VOCAB_FILE), 'rb') as fin:
            input_vocab = dill.load(fin)
        with open(os.path.join(path, cls.OUTPUT_VOCAB_FILE), 'rb') as fin:
            output_vocab = dill.load(fin)
        optimizer = start_epoch['optimizer']

        print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights,
                                                                         start_epoch['epoch']))
        return checkpoint(model=model, input_vocab=input_vocab,
                          output_vocab=output_vocab, optimizer=optimizer,path=path)