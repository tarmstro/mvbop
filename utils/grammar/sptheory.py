"""
http://arxiv.org/pdf/1306.3888.pdf
"""
from collections import Counter


def MK10(corpus):
    corplst = list(corpus)
    bigrams = Counter()
    for i in xrange(len(corplst)):
        print ''.join(corpus[i:i+2])
        bigrams[''.join(corpus[i:i+2])] += 1
    common, count = bigrams.most_common()[0]
    print common
    print count


def SNPR(corpus):
    pass


if __name__ == '__main__':
    demo = 'aabbccddaabbccddbbccbbccaabbccccddccdd'
    MK10(demo)
