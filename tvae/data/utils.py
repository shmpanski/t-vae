import csv

import numpy as np


class SentenceIterator:
    def __init__(self, data_file_name, spm):
        """
        Load and iterate dataset.
        Args:
            data_file_name (str): dataset file name.
            spm (sentencepice.SentencePieceProcessor): Sentencepiece model.
        """
        self.data_file_name = data_file_name
        self.spm = spm
        self.bos_id, self.eos_id = spm.bos_id(), spm.eos_id()

        with open(self.data_file_name) as file:
            reader = csv.reader(file)

            self.data = np.array(
                [[self.bos_id] + self.spm.EncodeAsIds(row[0]) + [self.eos_id] for row in reader])

    def __iter__(self):
        for row in self.data:
            yield list(map(str, row))

    def export(self, filename):
        """
        Export sentence data into .npy file.
        Args:
            filename (str): exported filename.
        """
        np.save(filename, self.data)


def export_embeddings(filename, sp_model, w2v_model):
    """Export embeddings into numpy matrix.
    Args:
        filename (str): the name of the exported file.
        sp_model (sentencepice.SentencePieceProcessor): Sentencepice model.
        w2v_model (gensim.models.Word2Vec): Word2Vec model.
    """
    dim = w2v_model.vector_size
    vocab_size = len(sp_model)
    table = np.array([
        w2v_model[str(i)] if str(i) in w2v_model.wv else np.zeros([dim])
        for i in range(vocab_size)
    ])
    np.save(filename, table)
