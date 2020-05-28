import numpy as np


class KMeans:
    def __init__(self):
        def __init__(self, k=3, num_iter=1000):
        """
            Parameter:
                k: Number of clusters we are trying to classify
                num_iter: Number of iterations we are going to loop
        """

        self.model_name = 'KMeans'
        self.k = k
        self.num_iter = num_iter
        self.centers = None
        self.RM = None

    def _cloest_centorids(arr, centers):
        distance = [np.sqrt(np.sum(np.square(X[j, :] - center))) for
                    center in centers]
        minpos = np.argmin(distance)
        return minpos

    def fit(self, X):
        """
            Train the given dataset

            Parameter:
                X: Matrix or 2-D array. Input feature matrix.

            Return:
                self: the whole model containing relevant information
        """

        r, c = X.shape
        centers = []
        RM = np.zeros((r, self.k))

        # initialize centers
        initials = np.random.choice(r, self.k)
        for i in initials:
            centers.append(X[i, :])
        centers = np.array(centers)

        for i in range(self.num_iter):
            for j in range(r):
                # calculate cloest centroids
                minpos = self._cloest_centorids(X[j, :], centers)

                temp_rm = np.zeros(self.k)
                temp_rm[minpos] = 1
                RM[j, :] = temp_rm
            new_centers = centers.copy()
            for l in range(self.k):
                # find cloest centers
                row_index = (RM[:, l] == 1).flatten()
                all_l = X[row_index, :]
                new_centers[l, :] = np.mean(all_l, axis=0)
            if np.sum(new_centers - centers) < 0.000000000000000000001:
                # update centers and RM
                self.centers = new_centers
                self.RM = RM
                return self
            centers = new_centers
        self.centers = centers
        self.RM = RM
        return self
