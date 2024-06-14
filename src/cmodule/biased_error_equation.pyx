import numpy as np
cimport numpy as np
from abc import ABCMeta, abstractmethod
import random

cdef class ErrorEquation:
    __metaclass__ = ABCMeta

    @abstractmethod
    def init_params(self, np.ndarray matrix, float mean, int k, float regularization_rate) -> None:
        pass

    @abstractmethod
    def update_params(self, float learning_rate) -> bool:
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def calc_error(self) -> float:
        pass

cdef class BiasedErrorEquation(ErrorEquation):
    cdef np.ndarray matrix
    cdef float mean
    cdef float regularization_rate
    cdef np.ndarray __error_matrix
    cdef np.ndarray __p
    cdef np.ndarray __q
    cdef np.ndarray __user_b
    cdef np.ndarray __movie_b

    def init_params(self, np.ndarray matrix, float mean, int k, float regularization_rate) -> None:
        cdef float sigma = np.sqrt(1 / k)
        cdef int axis1 = matrix.shape[0]
        cdef int axis2 = matrix.shape[1]
        self.matrix = matrix

        self.__p = np.random.normal(0, sigma, (axis1, k)).astype(np.float32)
        self.__q = np.random.normal(0, sigma, (k, axis2)).astype(np.float32)
        self.__user_b = np.zeros(axis1, dtype=np.float32)
        self.__movie_b = np.zeros(axis2, dtype=np.float32)
        self.mean = mean
        self.regularization_rate = regularization_rate

    def update_params(self, float learning_rate) -> bool:
        cdef np.ndarray error_matrix = np.full([self.matrix.shape[0], self.matrix.shape[1]], np.inf, dtype=np.float32)
        cdef int i, j

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if np.isinf(self.matrix[i, j]):
                    continue
                error_on_index = self.__calc_error_on_idx(i, j)
                self.__p[i] += learning_rate * self.__calc_p_diff_on_idx(i, j, error_on_index)
                self.__q[:, j] += learning_rate * self.__calc_q_diff_on_idx(i, j, error_on_index)
                self.__user_b[i] += learning_rate * self.__calc_user_bias_diff_on_idx(i, j, error_on_index)
                self.__movie_b[j] += learning_rate * self.__calc_movie_bias_diff_on_idx(i, j, error_on_index)
                error_matrix[i, j] = error_on_index
        self.__error_matrix = error_matrix

        return True

    def predict(self) -> np.ndarray:
        cdef np.ndarray result
        cdef int shape0 = self.matrix.shape[0]
        cdef int shape1 = self.matrix.shape[1]

        result = (
            self.__p @ self.__q
            + np.tile(self.__user_b, (shape1, 1)).T
            + np.tile(self.__movie_b, (shape0, 1))
            + self.mean
        )
        return result

    def calc_error(self) -> float:
        cdef float error_sq_sum = 0
        cdef int error_cnt = 0
        cdef int axis1 = self.__error_matrix.shape[0]
        cdef int axis2 = self.__error_matrix.shape[1]
        cdef int i, j

        for i in range(axis1):
            for j in range(axis2):
                if np.isinf(self.__error_matrix[i, j]):
                    continue
                error_sq_sum += self.__error_matrix[i, j] ** 2
                error_cnt += 1
        return np.sqrt(error_sq_sum / error_cnt)

    cdef float __calc_error_on_idx(self, int r, int c):
        return (
            self.matrix[r, c]
            - (
                np.dot(self.__p[r], self.__q[:, c])
                + self.__user_b[r]
                + self.__movie_b[c]
                + self.mean
            )
            + self.regularization_rate
            * (
                np.linalg.norm(self.__p[r], 2)
                + np.linalg.norm(self.__q[:, c], 2)
                + self.__user_b[r] ** 2
                + self.__movie_b[c] ** 2
            )
        )

    cdef np.ndarray __calc_p_diff_on_idx(self, int r, int c, float error_on_idx):
        return error_on_idx * self.__q[:, c] - self.regularization_rate * self.__p[r]

    cdef np.ndarray __calc_q_diff_on_idx(self, int r, int c, float error_on_idx):
        return error_on_idx * self.__p[r] - self.regularization_rate * self.__q[:, c]

    cdef float __calc_user_bias_diff_on_idx(self, int r, int c, float error_on_idx):
        return error_on_idx - self.regularization_rate * self.__user_b[r]

    cdef float __calc_movie_bias_diff_on_idx(self, int r, int c, float error_on_idx):
        return error_on_idx - self.regularization_rate * self.__movie_b[c]