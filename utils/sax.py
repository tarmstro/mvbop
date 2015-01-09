# Implementation of the SAX -- Symbolic Aggregate approXimation
#
#
import numpy as np
import string

class SAX(object):
    _normal_cutoffs = {
                        2: [0],
                        3: [-0.43, 0.43],
                        4: [-0.67, 0, 0.67],
                        5: [-0.84, -0.25, 0.25, 0.84],
                        6: [-0.97, -0.43, 0, 0.43, 0.97],
                        7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                        8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
                        9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
                        10: [-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
                        11: [-1.34, -0.91, -0.6, -0.35, -0.11, 0.11, 0.35, 0.6, 0.91, 1.34],
                        12: [-1.38, -0.97, -0.67, -0.43, -0.21, 0, 0.21, 0.43, 0.67, 0.97, 1.38],
                        13: [-1.43, -1.02, -0.74, -0.5, -0.29, -0.1, 0.1, 0.29, 0.5, 0.74, 1.02, 1.43],
                        14: [-1.47, -1.07, -0.79, -0.57, -0.37, -0.18, 0, 0.18, 0.37, 0.57, 0.79, 1.07, 1.47],
                        15: [-1.5, -1.11, -0.84, -0.62, -0.43, -0.25, -0.08, 0.08, 0.25, 0.43, 0.62, 0.84, 1.11, 1.5],
                        16: [-1.53, -1.15, -0.89, -0.67, -0.49, -0.32, -0.16, 0, 0.16, 0.32, 0.49, 0.67, 0.89, 1.15, 1.53],
                        17: [-1.56, -1.19, -0.93, -0.72, -0.54, -0.38, -0.22, -0.07, 0.07, 0.22, 0.38, 0.54, 0.72, 0.93, 1.19, 1.56],
                        18: [-1.59, -1.22, -0.97, -0.76, -0.59, -0.43, -0.28, -0.14, 0, 0.14, 0.28, 0.43, 0.59, 0.76, 0.97, 1.22, 1.59],
                        19: [-1.62, -1.25, -1, -0.8, -0.63, -0.48, -0.34, -0.2, -0.07, 0.07, 0.2, 0.34, 0.48, 0.63, 0.8, 1, 1.25, 1.62],
                        20: [-1.64, -1.28, -1.04, -0.84, -0.67, -0.52, -0.39, -0.25, -0.13, 0, 0.13, 0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64]
                    }
    _alphabet = string.ascii_lowercase[:19]
    default_points_per_symbol = 1
    default_a = 3


    def __init__(self, series, **kwargs):
        """SAXify the timeseries

            Args:
                series: a univariate timeseries in the form of a numpy array
                points_per_symbol: number of data points represented per SAX symbol
                a: alphabet size (see global normal_cutoffs) -- 2 through 20
        """
        self.series = series
        self.orig = series
        self.points_per_symbol = kwargs.get('points_per_symbol', 1)
        self.a = kwargs.get('a', 3)
        self._standardize()
        self._paa()
        self._sax = np.array([])
        for mean in self.paa:
            i = 0
            for cutoff in self._normal_cutoffs[self.a]:
                if mean > cutoff:
                    i = i + 1
            self._sax = np.append(self._sax, i)


    def _paa(self):
        """Piecewise aggregate approximation of a timeseries

        Args:
            series: a numpy array of a univariate timeseries
            points_per_symbol: number of data points represented per SAX symbol
        """
        self.paa = np.array([self.series[i * self.points_per_symbol : (i + 1) * self.points_per_symbol].mean() for i in range(len(self.series) / self.points_per_symbol)])


    def _standardize(self):
        """Normalize the timeseries to have a mean of 0 and
        standard deviation of 1.

        Args:
            series: a univariate timeseries
        """
        deviation = np.std(self.series)
        self.series = (self.series - np.mean(self.series)) / (deviation if deviation != 0 else 1)


    def sax(self):
        return self._sax


    def stringify(self):
        """Get the sax string

        Returns
            A string containing the SAX string
        """
        buf = ''
        for i in self._sax:
            buf += self._alphabet[int(i)]
        return buf


    def  __str__(self):
        return self.stringify()


    def __repr__(self):
        return self.stringify()


    def min_dist(self, sax_word_2):
        """The minimum distance between two SAX words

        Args:
            sax_word_2: numpy array SAX word

        Return:
            The MinDist distance metric value
        """
        dist = 0.0
        diffs = np.apply_along_axis(np.abs, 0, np.subtract(self.sax(), sax_word_2.sax()))
        for i in range(len(diffs)):
            if diffs[i] > 1:
                dist += self._normal_cutoffs[self.a][int(max(self.sax()[i], sax_word_2.sax()[i])) - 1]
                dist -= self._normal_cutoffs[self.a][int(min(self.sax()[i], sax_word_2.sax()[i]))]
        return dist


if __name__ == '__main__':
    x = SAX(np.array(range(0, 20)))
    y = SAX(np.array(range(20, 0, -1)))

    print x.sax()
    print x.stringify()
    print x.min_dist(y)
    print x.min_dist(x)

    print SAX(np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,6,6,6,6,10,100]), points_per_symbol=1, a=5).stringify()
    print SAX(np.array([7,1,4,4,4,4]), points_per_symbol=1, a=5).stringify()
