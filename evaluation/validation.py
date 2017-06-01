"""
This module holds classes related to validation.
"""


class LeaveOneOutCrossValidator:
    """
    Represents a leave-one-out cross-validation.
    """

    def __init__(self, n: int):
        """
        Initializes a new instance of the LeaveOneOutCrossValidator class.

        :param n: The number of samples.
        :type n: int
        """
        self.i = 0
        self.n = n

    def __iter__(self):
        """
        Gets an iterator object.

        :return: self
        :rtype: LeaveOneOutCrossValidator
        """
        return self

    def __next__(self):
        """
        Generates the next training and testing indices.

        :return: (train, test) the training and testing indices.
        """
        if self.i < self.n:
            test = self.i
            train = list(range(self.n))
            del train[test]
            self.i += 1
            return train, test
        else:
            raise StopIteration()

    def __str__(self):
        """
        Gets a nicely printable string representation.

        :return: String representation.
        """
        return 'LeaveOneOutCrossValidator:\n' \
               ' i: {self.i}\n' \
               ' n: {self.n}\n' \
            .format(self=self)
