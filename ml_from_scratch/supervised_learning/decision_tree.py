import numpy as np


class DecisionTree:
    """

    Decision Tree Classifier

    Attributes:
        root: Root Node of the tree.
        max_depth: Max depth allowed for the tree
        size_allowed : Min_size split, smallest size allowed for split
        n_features: Number of features to use during building the tree.(Random Forest)
        n_split:  Number of split for each feature. (Random Forest)

    """

    def __init__(self, max_depth=1000, size_allowed=1, n_features=None, n_split=None):
        """
            Initializations for class attributes.
        """

        self.root = 1
        self.max_depth = max_depth
        self.size_allowed = size_allowed
        self.n_features = n_features
        self.n_split = n_split

    class Node:
        """
            Node Class for the building the tree.

            Attribute:
                threshold: The threshold like if x1 < threshold, for spliting.
                feature: The index of feature on this current node.
                left: Pointer to the node on the left.
                right: Pointer to the node on the right.
                pure: Bool, describe if this node is pure.
                predict: Class, indicate what the most common Y on this node.

        """

        def __init__(self, threshold=None, feature=None):
            """
                Initializations for class attributes.
            """

            self.threshold = threshold
            self.feature = feature
            self.left = None
            self.right = None
            self.pure = False
            self.depth = 1
            self.predict = None

    def entropy(self, lst):
        """
            Function Calculate the entropy given lst.

            Attributes:
                entro: variable store entropy for each step.
                classes: all possible classes. (without repeating terms)
                counts: counts of each possible classes.
                total_counts: number of instances in this lst.

            lst is vector of labels.
        """

        entro = 0
        classes, counts = np.unique(lst, return_counts=True)
        total_counts = len(lst)
        probs = counts / total_counts
        for i in probs:
            # ignore prob with 0
            if i != 0:
                entro = entro - i * np.log(i)
        return entro

    def information_gain(self, lst, values, threshold):
        """

            Function Calculate the information gain, by using entropy function.

            lst is vector of labels.
            values is vector of values for individule feature.
            threshold is the split threshold we want to use for calculating the entropy.

        """
        # find the left and right indices
        _less_or_equal = np.where(values <= threshold)[0]
        _above = np.where(values > threshold)[0]

        left_prop = len(_less_or_equal) / len(values)
        right_prop = len(_above) / len(values)

        left_entropy = self.entropy(_less_or_equal)
        right_entropy = self.entropy(_above)

        return self.entropy(lst) - (
            left_prop * left_entropy + right_prop * right_entropy
        )

    def find_rules(self, data):
        """

            Helper function to find the split rules.

            data is a matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.

        """
        n, m = 1, 1
        rules = []
        for i in data.T:
            unique_value = np.unique(i)
            # get the midpoint between each unique addjaacent value
            diff = [
                (unique_value[x] + unique_value[x + 1]) / 2
                for x in range(len(unique_value) - 1)
            ]
            rules.append(diff)
        return rules

    def next_split(self, data, label):
        """
            Helper function to find the split with most information gain, 
            by using find_rules, and information gain.

            data is a matrix or 2-D numpy array, represnting training instances.
            Each training instance is a feature vector.

            label contains the corresponding labels. 
        """

        rules = self.find_rules(data)
        max_info = -float("inf")
        num_col = None
        threshold = None

        """
            Check Number of features to use, None means all features. (Decision Tree always use all feature)
            If n_features is a int, use n_features of features by random choice.
            If n_features == 'sqrt', use sqrt(Total Number of Features ) by random choice
        """
        if not self.n_features:
            index_col = np.arange(data.shape[1])
        else:
            if self.n_features == "sqrt":
                num_index = int(np.sqrt(data.shape[1]))
            elif isinstance(self.n_features, int):
                num_index = self.n_features
            np.random.seed()
            index_col = np.random.choice(data.shape[1], num_index, replace=False)

        """
            Do the similar selection we did for features, n_split take in None or int or 'sqrt'.
            For all selected feature and corresponding rules, we check it's information gain.
        """
        _data_T = data.T
        for i in index_col:
            count_temp_rules = len(rules[i])
            if not self.n_split:
                index_rules = np.arange(count_temp_rules)
            else:
                if self.n_split == "sqrt":
                    num_rules = int(np.sqrt(len(count_temp_rules)))
                elif isinstance(self.n_split, int):
                    num_rules = self.n_split
                np.random.seed()
                # get partial indices in a rule list
                index_rules = np.random.choice(
                    count_temp_rules, num_rules, replace=False
                )

            for j in index_rules:
                info = self.information_gain(label, _data_T[i], rules[i][j])
                if info > max_info:
                    max_info = info
                    num_col = i
                    threshold = rules[i][j]
        return threshold, num_col

    def build_tree(self, X, y, depth):
        """
            Helper function for building the tree.
        """

        first_threshold, first_feature = self.next_split(X, y)
        current = self.Node(first_threshold, first_feature)
        self.root = current

        """
        Check if we pass the max_depth, check if the first_feature is None, min split size.
        If some of those condition met, change current to pure, and set predict to the most popular label
            and return current
        """
        if (
            depth > self.max_depth
            or first_feature == None
            or X.shape[0] <= self.size_allowed
        ):
            _values, _counts = np.unique(y, return_counts=True)
            ind = np.argmax(_counts)
            current.predict = _values[ind]
            current.pure = True
            return current

        # Check if there is only 1 label in this node, change current to pure, and set predict to the label
        if len(np.unique(y)) == 1:
            current.predict = y[0]
            current.pure = True
            return current

        # Find the left node index with feature i <= threshold  Right with feature i > threshold.
        left_index = X[:, first_feature] <= first_threshold
        right_index = X[:, first_feature] > first_threshold

        # If we either side is empty, change current to pure, and set predict to the label
        if len(left_index) == 0 or len(right_index) == 0:
            _values, _counts = np.unique(y, return_counts=True)
            ind = np.argmax(_counts)
            current.predict = _values[ind]
            current.pure = True
            return current

        left_X, left_y = X[left_index, :], y[left_index]
        current.left = self.build_tree(left_X, left_y, depth + 1)

        right_X, right_y = X[right_index, :], y[right_index]
        current.right = self.build_tree(right_X, right_y, depth + 1)

        return current

    def fit(self, X, y):
        """
            The fit function fits the Decision Tree model based on the training data. 

            X_train is a matrix or 2-D numpy array, represnting training instances. 
            Each training instance is a feature vector. 

            y_train contains the corresponding labels. There might be multiple (i.e., > 2) classes.
        """
        self.root = self.build_tree(X, y, 1)
        return self

    def ind_predict(self, inp):
        """
            Predict the most likely class label of one test instance based on its feature vector x.
        """
        cur = self.root
        # Stop condition we are at a node is pure.
        while not cur.pure:
            feature = cur.feature
            threshold = cur.threshold
            if inp[feature] <= threshold:
                cur = cur.left
            else:
                cur = cur.right
        return cur.predict

    def predict(self, inp):
        """
            X is a matrix or 2-D numpy array, represnting testing instances. 
            Each testing instance is a feature vector. 

            Return the predictions of all instances in a list.
        """

        result = [self.ind_predict(inp[i]) for i in inp.shape[0]]
        return result
