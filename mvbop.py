from utils.sax import Sax

def main():
    x = np.random.rand(50) * 20
    y = np.random.rand(50) * 20
    print _standardize(x)
    print sax(x, 5, 10)
    print min_dist(sax(x, 5, 10), sax(y, 5, 10), 10)
    print min_dist(sax(x, 5, 10), sax(x, 5, 10), 10)
    print sax_stringify(sax(x, 5, 10), 10)

    ### Tests

    print sax_stringify(sax([7,1,4,4,4,4], 1, 5), 5)
    print sax_stringify(sax([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6,6,6,6,10,100], 1, 5), 5)

