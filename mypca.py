import numpy as np


class SimplePCA:
    def __init__(self, n_components=2):
        """SimplePCA initialization.

        Parameters
        ----------
        n_components : int, default = 2
        Number of principal components

        """
        self.num_components = n_components
        self.x_std = np.array([0])

    def fit_transform(self, X):
        """Fit and the model with X and apply the dimension reduction in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        Returns
        -------
        project :  ndarray of shape (n_samples, n_components)
            Returns the transformed X (PCs)
        """
        # Update variable x_std with the standardization
        self.get_standardization(X)
        # Apply the PCA steps
        explanation, pcs = self.get_eigen()
        project = self.project_matrix(pcs)
        self.get_variance(project)

        return project

    def get_covariant(self):
        """Calculated the covariant matrix of X.

        Parameters
        ----------
        self : object
        Returns
        -------
        covariant :  ndarray of shape (n_samples, n_samples)
            Returns a square covariant matrix
        """
        covariant = np.dot(self.x_std.T, self.x_std)
        return covariant

    def get_standardization(self, X, stand=False):
        """Centralize and standize the data.
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The data input.
        stand: boolean, default=False
            If True, standarize the input data.
        Returns
        -------
        covariant :  ndarray of shape (n_samples, n_samples)
            Returns a square covariant matrix
        """
        mean = np.mean(X, axis=0)
        if stand:
            std = np.std(X, axis=0)
            x_std = (X - mean)/ std
        else:
            x_std = (X - mean)
        self.x_std = x_std

    def get_eigen(self):
        """Calculate the eigenvalue and eigenvectors
        Parameters
        ----------
        self : object.
        Returns
        -------
        explanation :  ndarray of shape (n_components,)
            Ordered variance
         pcs: ndarray of shape (n_components, n_features)
            Principal components
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.get_covariant())

        projection = self.project_matrix(eigenvectors)
        variance = self.get_variance(projection)
        # Order based on the variance
        v_components = np.argsort(variance)[::-1][:self.num_components]
        # The principal components are the transposed eigenvectors.
        pcs = eigenvectors[:, v_components].T
        explanation = variance[v_components]
        return explanation, pcs

    def project_matrix(self, eigenvectors):
        """Calculate the projection of X in the eigenvectors"""
        return np.dot(self.x_std, eigenvectors.T)

    def get_variance(self, projection):
        """Calculate variance"""
        variance = np.var(projection, axis=0, ddof=1)
        return variance
