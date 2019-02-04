import unittest
import tempfile
import os

from tvae.data import TVAEDataset


class TestBPEDatasetMethods(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.prefix = "test"
        self.part = "train"
        self.dev_part = "dev"
        self.workdir = os.path.join(self.directory.name, self.prefix)

        train_source_file_name = os.path.join(self.directory.name, "train.tsv")
        dev_source_file_name = os.path.join(self.directory.name, "dev.tsv")
        with open(train_source_file_name, "w+") as train_file:
            train_file.writelines(["I have writen some code\r\n",
                                   "There is an apple on the table\r\n",
                                   "I like cats and dogs\r\n"])
        with open(dev_source_file_name, "w+") as dev_file:
            dev_file.writelines(["Hello... It's me...\r\n",
                                 "I hope this thing ... works\r\n"
                                 "INTERESTING\r\n",
                                 "I have no imagination\r\n",
                                 "Enough\r\n"
                                 ])

    def tearDown(self):
        self.directory.cleanup()

    def test_init(self):
        dataset = TVAEDataset(self.directory.name, self.prefix, self.part,
                              vocab_size=30, embedding_size=64, max_sequence_length=150)
        exist = TVAEDataset.exist(self.directory.name, self.prefix, "train")
        self.assertEqual(len(dataset), 3)
        self.assertTrue(exist)

    def test_init_multiple_parts(self):
        train_dataset = TVAEDataset(self.directory.name, self.prefix, self.part,
                                    vocab_size=30, embedding_size=64, max_sequence_length=150)
        dev_dataset = TVAEDataset(self.directory.name, self.prefix, self.dev_part,
                                  spm_model=train_dataset.spm_model, max_sequence_length=150)
        exist = TVAEDataset.exist(self.directory.name, self.prefix, ["train", "dev"])
        self.assertEqual(len(dev_dataset), 5)
        self.assertTrue(exist)

    def test_get_embeddings(self):
        dataset = TVAEDataset(self.directory.name, self.prefix, self.part,
                              vocab_size=30, embedding_size=64, max_sequence_length=150)
        embeddings = dataset.get_embeddings()
        self.assertTupleEqual(embeddings.shape, (30, 64))
