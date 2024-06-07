import numpy as np
from abc import ABCMeta, abstractmethod
from movie_rating import RatingMatrix
import random


class ErrorEquation(metaclass=ABCMeta):
    @abstractmethod
    def init_params(
        self, matrix: RatingMatrix, k: int, regularization_rate: float
    ) -> None:
        pass

    @abstractmethod
    def update_params(self, learning_rate: float) -> bool:
        pass

    @abstractmethod
    def predict(self) -> np.ndarray:
        pass

    @abstractmethod
    def calc_error(self) -> float:
        pass


class BasicErrorEquation(ErrorEquation):
    matrix: np.ndarray
    mean: float
    regularization_rate: float

    __error_matrix: np.ndarray
    __p: np.ndarray
    __q: np.ndarray

    def init_params(
        self,
        matrix: RatingMatrix,
        k: int,
        regularization_rate: float,
    ) -> None:
        cent_val = np.sqrt(matrix.total_mean / k)
        sigma = np.sqrt(1 / k)
        axis1 = matrix.matrix.shape[0]
        axis2 = matrix.matrix.shape[1]
        self.matrix = matrix.matrix
        # self.__p = np.random.normal(cent_val, sigma, (axis1, k))
        # self.__q = np.random.normal(cent_val, sigma, (k, axis2))
        self.__p = np.full((axis1, k), random.normalvariate(cent_val, sigma))
        self.__q = np.full((k, axis2), random.normalvariate(cent_val, sigma))
        self.mean = matrix.total_mean
        self.regularization_rate = regularization_rate

    def update_params(self, learning_rate: float) -> bool:
        error_matrix = np.full(self.matrix.shape, np.inf)

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if np.isinf(self.matrix[i][j]):
                    continue
                error_on_index = self.__calc_error_on_idx(i, j)
                self.__p[i] += learning_rate * self.__calc_p_diff_on_idx(
                    i, j, error_on_index
                )
                self.__q[:, j] += learning_rate * self.__calc_q_diff_on_idx(
                    i, j, error_on_index
                )
                error_matrix[i][j] = error_on_index
        self.__error_matrix = error_matrix

        return True

    def predict(self) -> np.ndarray:
        return self.__p @ self.__q

    def calc_error(self) -> float:
        error_sq_sum = 0
        error_cnt = 0
        axis1 = self.__error_matrix.shape[0]
        axis2 = self.__error_matrix.shape[1]
        for i in range(axis1):
            for j in range(axis2):
                if np.isinf(self.__error_matrix[i][j]):
                    continue
                error_sq_sum += self.__error_matrix[i][j] ** 2
                error_cnt += 1
        return np.sqrt(error_sq_sum / error_cnt)

    def __calc_error_on_idx(self, r: int, c: int) -> float:
        return (
            self.matrix[r][c]
            - self.__p[r] @ self.__q[:, c]
            + self.regularization_rate
            * (np.linalg.norm(self.__p[r], 2) + np.linalg.norm(self.__q[:, c], 2))
        )

    def __calc_p_diff_on_idx(self, r: int, c: int, error_on_idx: float) -> np.ndarray:
        return error_on_idx * self.__q[:, c] - self.regularization_rate * self.__p[r]

    def __calc_q_diff_on_idx(self, r: int, c: int, error_on_idx: float) -> np.ndarray:
        return error_on_idx * self.__p[r] - self.regularization_rate * self.__q[:, c]


class BiasedErrorEquation(ErrorEquation):
    matrix: np.ndarray
    mean: float
    regularization_rate: float

    __error_matrix: np.ndarray
    __p: np.ndarray
    __q: np.ndarray
    __user_b: np.ndarray
    __movie_b: np.ndarray

    def init_params(
        self,
        matrix: RatingMatrix,
        k: int,
        regularization_rate: float,
    ) -> None:
        sigma = np.sqrt(1 / k)
        axis1 = matrix.matrix.shape[0]
        axis2 = matrix.matrix.shape[1]
        self.matrix = matrix.matrix
        # self.__p = np.random.normal(0, sigma, (axis1, k))
        # self.__q = np.random.normal(0, sigma, (k, axis2))
        self.__p = np.full((axis1, k), random.normalvariate(0, sigma))
        self.__q = np.full((k, axis2), random.normalvariate(0, sigma))
        self.__user_b = np.zeros(axis1)
        self.__movie_b = np.zeros(axis2)
        self.mean = matrix.total_mean
        self.regularization_rate = regularization_rate

    def update_params(self, learning_rate: float) -> bool:
        error_matrix = np.full(self.matrix.shape, np.inf)

        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                if np.isinf(self.matrix[i][j]):
                    continue
                error_on_index = self.__calc_error_on_idx(i, j)
                self.__p[i] += learning_rate * self.__calc_p_diff_on_idx(
                    i, j, error_on_index
                )
                self.__q[:, j] += learning_rate * self.__calc_q_diff_on_idx(
                    i, j, error_on_index
                )
                self.__user_b[i] += learning_rate * self.__calc_user_bias_diff_on_idx(
                    i, j, error_on_index
                )
                self.__movie_b[j] += learning_rate * self.__calc_movie_bias_diff_on_idx(
                    i, j, error_on_index
                )
                error_matrix[i][j] = error_on_index
        self.__error_matrix = error_matrix

        return True

    def predict(self) -> np.ndarray:
        shape = self.matrix.shape
        print(shape)
        print("p: ", self.__p.shape)
        print("q: ", self.__q.shape)
        print("user_b: ", self.__user_b.shape)
        print("tile user_b: ", np.tile(self.__user_b, (shape[1], 1)).shape)
        print("movie_b: ", self.__movie_b.shape)
        print("tile movie_b: ", np.tile(self.__movie_b, (shape[0], 1)).shape)
        return (
            self.__p @ self.__q
            + np.tile(self.__user_b, (shape[1], 1)).T
            + np.tile(self.__movie_b, (shape[0], 1))
            + self.mean
        )

    def calc_error(self) -> float:
        error_sq_sum = 0
        error_cnt = 0
        axis1 = self.__error_matrix.shape[0]
        axis2 = self.__error_matrix.shape[1]
        for i in range(axis1):
            for j in range(axis2):
                if np.isinf(self.__error_matrix[i][j]):
                    continue
                error_sq_sum += self.__error_matrix[i][j] ** 2
                error_cnt += 1
        return np.sqrt(error_sq_sum / error_cnt)

    def __calc_error_on_idx(self, r: int, c: int) -> float:
        return (
            self.matrix[r][c]
            - (
                self.__p[r] @ self.__q[:, c]
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

    def __calc_p_diff_on_idx(self, r: int, c: int, error_on_idx: float) -> np.ndarray:
        return error_on_idx * self.__q[:, c] - self.regularization_rate * self.__p[r]

    def __calc_q_diff_on_idx(self, r: int, c: int, error_on_idx: float) -> np.ndarray:
        return error_on_idx * self.__p[r] - self.regularization_rate * self.__q[:, c]

    def __calc_user_bias_diff_on_idx(
        self, r: int, c: int, error_on_idx: float
    ) -> float:
        return error_on_idx - self.regularization_rate * self.__user_b[r]

    def __calc_movie_bias_diff_on_idx(
        self, r: int, c: int, error_on_idx: float
    ) -> float:
        return error_on_idx - self.regularization_rate * self.__movie_b[c]
