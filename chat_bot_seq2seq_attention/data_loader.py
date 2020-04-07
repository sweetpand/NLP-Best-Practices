import os
import nltk
import numpy as np
import pickle
import random


padToken, goToken, eosToken, unknownToken = 0, 1, 2, 3

class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []



def loadDataset(filename):
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        word2id = data['word2id']
        id2word = data['id2word']
        trainingSamples = data['trainingSamples']
    print(word2id)
    print(id2word)
    print(trainingSamples)
    return word2id, id2word, trainingSamples

def createBatch(samples):
    batch = Batch()
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for sample in samples:
        source = list(reversed(sample[0]))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)
        target = sample[1]
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)
        #batch.target_inputs.append([goToken] + target + pad[:-1])

    return batch

def getBatches(data, batch_size):
    random.shuffle(data)
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches

def sentence2enco(sentence, word2id):
    if sentence == '':
        return None
    tokens = nltk.word_tokenize(sentence)
    if len(tokens) > 20:
        return None
    wordIds = []
    for token in tokens:
        wordIds.append(word2id.get(token, unknownToken))
    batch = createBatch([[wordIds, []]])
    return batch
