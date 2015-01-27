"""
http://arxiv.org/pdf/1306.3888.pdf
"""
from collections import Counter


def MK10(corpus):
    corpus = '_'.join(corpus)
    bigrams = Counter()
    for i in xrange(0, len(corpus) - 1, 2):
        print corpus[i:i+4]
        bigrams[corpus[i:i+4]] += 1
    common, count = bigrams.most_common()[0]
    print common


def SNPR(corpus):
    pass


if __name__ == '__main__':
    demo = 'aabbccddaabbccddbbccbbccaabbccccddccdd'
    MK10(demo)
