"""This module holds classes related to validation."""


class LeaveOneOutCrossValidator:
    """Represents a leave-one-out cross-validation."""

    def __init__(self, n: int):
        """Initializes a new instance of the LeaveOneOutCrossValidator class.

        Args:
            n (int): The number of samples.
        """
        self.i = 0
        self.n = n

    def __iter__(self):
        """Gets an iterator object.

        Returns:
            LeaveOneOutCrossValidator: Itself.
        """
        return self

    def __next__(self):
        """Generates the next training and testing indices.

        Returns:
            (list of int, list of int): The training and testing indices, respectively.
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
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        return 'LeaveOneOutCrossValidator:\n' \
               ' i: {self.i}\n' \
               ' n: {self.n}\n' \
            .format(self=self)
