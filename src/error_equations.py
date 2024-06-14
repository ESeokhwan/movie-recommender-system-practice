import numpy as np
from abc import ABCMeta, abstractmethod
import random


class ErrorEquation(metaclass=ABCMeta):
    @abstractmethod
    def init_params(
        self, matrix: np.ndarray, mean: float, k: int, regularization_rate: float
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
        matrix: np.ndarray,
        mean: float,
        k: int,
        regularization_rate: float,
    ) -> None:
        cent_val = np.sqrt(mean / k)
        sigma = np.sqrt(1 / k)
        axis1 = matrix.shape[0]
        axis2 = matrix.shape[1]
        self.matrix = matrix
        self.__p = np.full((axis1, k), random.normalvariate(cent_val, sigma))
        self.__q = np.full((k, axis2), random.normalvariate(cent_val, sigma))
        self.mean = mean
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
        matrix: np.ndarray,
        mean: float,
        k: int,
        regularization_rate: float,
    ) -> None:
        sigma = np.sqrt(1 / k)
        axis1 = matrix.shape[0]
        axis2 = matrix.shape[1]
        self.matrix = matrix

        init_p = random.normalvariate(0, sigma)
        init_q = random.normalvariate(0, sigma)
        print(f"init_p: {init_p:.4f}, init_q: {init_q:.4f}")
        self.__p = np.full((axis1, k), init_p)
        self.__q = np.full((k, axis2), init_q)
        # self.__p = np.zeros((axis1, k))
        # self.__q = np.zeros((k, axis2))
        self.__user_b = np.zeros(axis1)
        self.__movie_b = np.zeros(axis2)
        self.mean = mean
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


class BiasedPPErrorEquation(ErrorEquation):
    matrix: np.ndarray
    mean: float
    regularization_rate: float

    __rated_matrix: np.ndarray
    __error_matrix: np.ndarray
    __p: np.ndarray
    __q: np.ndarray
    __user_b: np.ndarray
    __movie_b: np.ndarray
    __y: np.ndarray

    def init_params(
        self,
        matrix: np.ndarray,
        mean: float,
        k: int,
        regularization_rate: float,
    ) -> None:
        sigma = np.sqrt(1 / k)
        axis1 = matrix.shape[0]
        axis2 = matrix.shape[1]
        self.matrix = matrix
        self.__rated_matrix = BiasedPPErrorEquation.__get_rated_matrix(matrix)

        init_p = random.normalvariate(0, sigma)
        init_q = random.normalvariate(0, sigma)
        print(f"init_p: {init_p:.4f}, init_q: {init_q:.4f}")
        self.__p = np.full((axis1, k), init_p)
        self.__q = np.full((k, axis2), init_q)
        # self.__p = np.zeros((axis1, k))
        # self.__q = np.zeros((k, axis2))
        self.__user_b = np.zeros(axis1)
        self.__movie_b = np.zeros(axis2)
        self.__y = np.zeros((axis2, k))
        self.mean = mean
        self.regularization_rate = regularization_rate

    def update_params(self, learning_rate: float) -> bool:
        error_matrix = np.full(self.matrix.shape, np.inf)

        for i in range(self.matrix.shape[0]):
            rm_norm = np.linalg.norm(self.__rated_matrix[i], 0.5)
            addition_on_p = self.__calc_addition_on_p(i)
            for j in range(self.matrix.shape[1]):
                if np.isinf(self.matrix[i][j]):
                    continue
                error_on_index = self.__calc_error_on_idx(i, j, rm_norm, addition_on_p)
                self.__p[i] += learning_rate * self.__calc_p_diff_on_idx(
                    i, j, error_on_index
                )
                self.__q[:, j] += learning_rate * self.__calc_q_diff_on_idx(
                    i, j, error_on_index, rm_norm, addition_on_p
                )
                self.__user_b[i] += learning_rate * self.__calc_user_bias_diff_on_idx(
                    i, j, error_on_index
                )
                self.__movie_b[j] += learning_rate * self.__calc_movie_bias_diff_on_idx(
                    i, j, error_on_index
                )
                for w in range(self.matrix.shape[1]):
                    if self.__rated_matrix[i][w] == 0:
                        continue
                    self.__y[w] += learning_rate * self.__calc_y_diff_on_idx(
                        i, j, w, error_on_index, rm_norm
                    )
                error_matrix[i][j] = error_on_index
        self.__error_matrix = error_matrix

        return True

    def predict(self) -> np.ndarray:
        shape = self.matrix.shape
        return (
            (self.__p + self.__calc_addition_matrix_on_p()) @ self.__q
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

    def __calc_addition_matrix_on_p(self) -> np.ndarray:
        addition = np.zeros(self.__p.shape)
        for i in range(self.matrix.shape[0]):
            addition[i] = self.__calc_addition_on_p(i)
        return addition

    def __calc_error_on_idx(
        self, r: int, c: int, rm_norm: float, addition_on_p: np.ndarray
    ) -> float:
        return (
            self.matrix[r][c]
            - (
                (self.__p[r] + (1 / rm_norm) * addition_on_p) @ self.__q[:, c]
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
                + np.linalg.norm(self.__y[c], 2)
            )
        )

    def __calc_addition_on_p(self, r: int) -> np.ndarray:
        addition = np.zeros(self.__p[r].shape)
        for c in range(self.matrix.shape[1]):
            if np.isinf(self.matrix[r][c]):
                continue
            addition += self.__y[c]
        return addition

    def __calc_p_diff_on_idx(self, r: int, c: int, error_on_idx: float) -> np.ndarray:
        return error_on_idx * self.__q[:, c] - self.regularization_rate * self.__p[r]

    def __calc_q_diff_on_idx(
        self,
        r: int,
        c: int,
        error_on_idx: float,
        rm_norm: float,
        addition_on_p: np.ndarray,
    ) -> np.ndarray:
        return (
            error_on_idx * (self.__p[r] + (1 / rm_norm) * addition_on_p)
            - self.regularization_rate * self.__q[:, c]
        )

    def __calc_user_bias_diff_on_idx(
        self, r: int, c: int, error_on_idx: float
    ) -> float:
        return error_on_idx - self.regularization_rate * self.__user_b[r]

    def __calc_movie_bias_diff_on_idx(
        self, r: int, c: int, error_on_idx: float
    ) -> float:
        return error_on_idx - self.regularization_rate * self.__movie_b[c]

    def __calc_y_diff_on_idx(
        self, r: int, c: int, w: int, error_on_idx: float, rm_norm: float
    ) -> np.ndarray:
        if self.__rated_matrix[r][w] == 0:
            return np.zeros(self.__y[c].shape)
        return (
            error_on_idx * (1 / rm_norm) * self.__q[:, c]
            - self.regularization_rate * self.__y[w]
        )

    @staticmethod
    def __get_rated_matrix(matrix: np.ndarray) -> None:
        rated_matrix = np.full(matrix.shape, 1)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isinf(matrix[i][j]):
                    rated_matrix[i][j] = 0
        return rated_matrix
