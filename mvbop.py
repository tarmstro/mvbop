from utils.sax import SAX
import numpy as np

def main():
    x = SAX(np.array(range(0, 20)))
    y = SAX(np.array(range(20, 0, -1)))

    print x.sax()
    print x.stringify()
    print x.min_dist(y)
    print x.min_dist(x)


    print SAX(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6,6,6,6,10,100]), 1, 5).stringify()
    print SAX(np.array([7,1,4,4,4,4]), 1, 5).stringify()




if __name__ == '__main__':
    main()
