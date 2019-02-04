import errno
import logging
import os
from functools import reduce
from itertools import takewhile
from operator import and_
from typing import List, TypeVar

import numpy as np
import torch
from gensim.models import Word2Vec
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from torch.utils.data import Dataset

from tvae.data.utils import SentenceIterator, export_embeddings


class TVAEDataset(Dataset):
    def __init__(self, root, prefix, part, max_sequence_length=150, **kwargs):
        self.root = root
        self.prefix = prefix
        self.preprocess_args = kwargs

        data_file_name = os.path.join(root, prefix, part + ".npy")
        spm_file_name = os.path.join(root, prefix, "spm.model")

        if not TVAEDataset.exist(root, prefix, part):
            logging.info("Start preprocessing %s/%s/%s dataset", root, prefix, part)
            self.preprocess(root, prefix, part, **kwargs)

        if 'spm_model' in self.preprocess_args:
            logging.info("Use existed sentencepiece model")
            self.spm_model = self.preprocess_args['spm_model']
        else:
            logging.info("Load sentencepiece model from disk")
            self.spm_model = SentencePieceProcessor()
            self.spm_model.load(spm_file_name)

        self._data = np.load(data_file_name)

        self.pad_symbol = self.spm_model.pad_id()
        self.eos_symbol = self.spm_model.eos_id()

        self._len = self._data.shape[0]
        self.limit = max_sequence_length

        sequence_lens = [
            len(seq) for seq in self._data
        ]
        self.max_sequence_length = min(self.limit, max(sequence_lens))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return self._len

    def preprocess(self, directory: str, prefix: str, part: str, spm_model: SentencePieceProcessor = None,
                   pretrain_emb=True, vocab_size=3000, embedding_size=600,
                   max_sentence_length=16384, workers=3, skip_gramm=False):

        # Check data files existing
        workdir = os.path.join(directory, prefix)
        os.makedirs(workdir, exist_ok=True)

        data_part_file = os.path.join(directory, part + ".tsv")
        if not os.path.exists(data_part_file):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), data_part_file)

        if part not in ['train', 'develop']:
            assert spm_model is not None, "For non train part, `spm_model` must be specified."
        else:
            # Train sentecepiece:
            logging.info("Start training sentecepiece")
            spm_directory = os.path.join(workdir, "spm")
            spm_params = (
                "--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
                "--input={} --model_prefix={} --vocab_size={} --max_sentence_length={}".format(
                    data_part_file, spm_directory, vocab_size, max_sentence_length
                )
            )
            SentencePieceTrainer.Train(spm_params)
            spm_model = SentencePieceProcessor()
            spm_model.load(spm_directory + ".model")

            if pretrain_emb:
                # Train word2vec
                logging.info("Start training word2vec")
                train_senteces = SentenceIterator(data_part_file, spm_model)
                logging.info("Loaded train sentences")
                w2v_model = Word2Vec(train_senteces, min_count=0, workers=workers,
                                     size=embedding_size, sg=int(skip_gramm))
                w2v_model_filename = os.path.join(workdir, "word2vec.model")
                w2v_model.save(w2v_model_filename)

                # Export embeddings
                logging.info("Export embeddings")
                embeddings_filename = os.path.join(workdir, "embedding.npy")
                export_embeddings(embeddings_filename, spm_model, w2v_model)
                logging.info("Embeddings have been saved into {}".format(embeddings_filename))

        logging.info("Start exporting data file")
        source_file_name = os.path.join(directory, part + ".tsv")
        exported_file_name = os.path.join(workdir, part + ".npy")
        sentence_iterator = SentenceIterator(source_file_name, spm_model)
        sentence_iterator.export(exported_file_name)
        logging.info("{} exported".format(exported_file_name))
        logging.info("Data preprocessing completed")

    @staticmethod
    def exist(root: str, prefix: str, parts: TypeVar("P", str, List[str])) -> bool:
        if isinstance(parts, str):
            parts = [parts]
        parts_file_name = [os.path.join(root, prefix, part + ".npy") for part in parts]
        smp_file_name = os.path.join(root, prefix, "spm.model")

        necessary_files = parts_file_name + [smp_file_name]
        existing = [os.path.exists(filename) for filename in necessary_files]
        return reduce(and_, existing)

    @staticmethod
    def _pad_sequence(sequences, pad_symbol=0):
        sequence_lengths = [len(sequence) for sequence in sequences]
        max_len = max(sequence_lengths)
        for i, length in enumerate(sequence_lengths):
            to_add = max_len - length
            sequences[i] += [pad_symbol] * to_add
        return sequences, sequence_lengths

    def collate_function(self, batch):
        src_list, src_length_list = TVAEDataset._pad_sequence(
            [example[:self.limit] for example in batch], self.pad_symbol)
        batch = {
            "src": torch.LongTensor(src_list)
        }
        return batch

    def get_embeddings(self):
        """Load pretrain embeddings.
        Returns:
            np.array: Array with word2vec embeddings if this one exists, otherwise `None`.
        """

        embedinds_path = os.path.join(self.root, self.prefix, "embedding.npy")
        if not os.path.exists(embedinds_path):
            logging.info("Embedding file does not founded")
            return None
        else:
            logging.info("Loading embedding dump file")
            return np.load(embedinds_path)

    def decode(self, sequences):
        sequences = [list(takewhile(lambda x: x != self.eos_symbol, sequence)) for sequence in sequences]
        return [self.spm_model.DecodeIds([token.item() for token in sentence])
                for sentence in sequences]
