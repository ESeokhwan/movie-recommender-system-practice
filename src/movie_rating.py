import numpy as np
from dataclasses import dataclass, field, InitVar


class MovieRating:
    def __init__(self, user_id: str, movie_id: str, rating: float, time_stamp: int):
        self.user_id = user_id
        self.movie_id = movie_id
        self.rating = rating
        self.time_stamp = time_stamp

    def __str__(self):
        return f"{self.user_id}\t{self.movie_id}\t{self.rating}\t{self.time_stamp}"

    def __repr__(self):
        return (
            f"MovieRating({self.user_id}, {self.movie_id}, {self.rating},"
            f" {self.time_stamp})"
        )


@dataclass(frozen=True)
class RatingMatrix:
    __data_stream: InitVar[iter]

    matrix: np.ndarray = field(init=False)
    user_idx_map: dict = field(init=False)
    movie_idx_map: dict = field(init=False)
    total_mean: float = field(init=False)
    user_mean: np.ndarray = field(init=False)
    movie_mean: np.ndarray = field(init=False)

    def __post_init__(self, __data_stream: iter) -> None:
        matrix, user_idx_map, movie_idx_map = self.__generate_matrix(__data_stream)
        total_mean = self.__calc_total_mean(matrix)
        user_mean = self.__calc_mean_along_axis(matrix, 1)
        movie_mean = self.__calc_mean_along_axis(matrix, 0)

        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "user_idx_map", user_idx_map)
        object.__setattr__(self, "movie_idx_map", movie_idx_map)
        object.__setattr__(self, "total_mean", total_mean)
        object.__setattr__(self, "user_mean", user_mean)
        object.__setattr__(self, "movie_mean", movie_mean)

    def get_filled_matrix(self) -> np.ndarray:
        filled_matrix = self.matrix.copy()
        self.__fill_matrix_with_mean_by_axis(
            filled_matrix, self.user_mean, 0
        )  # user mean
        self.__fill_matrix_with_mean_by_axis(
            filled_matrix, self.movie_mean, 1
        )  # movie mean
        self.__fill_matrix_with_mean(filled_matrix, self.total_mean)  # total mean
        return filled_matrix

    @classmethod
    def __generate_matrix(cls, data: iter):
        training_data = {}

        user_idx_map_i = ({}, 0)
        movie_idx_map_i = ({}, 0)

        for record in data:
            user_idx_map_i = cls.__construct_user_idx_map(record, user_idx_map_i)
            movie_idx_map_i = cls.__construct_movie_idx_map(record, movie_idx_map_i)
            training_data = cls.__construct_rating_map(record, training_data)

        user_idx_map = user_idx_map_i[0]
        movie_idx_map = movie_idx_map_i[0]

        matrix = np.full((user_idx_map.__len__(), movie_idx_map.__len__()), np.inf)
        cls.__fill_matrix(matrix, training_data, user_idx_map, movie_idx_map)

        return (
            matrix,
            user_idx_map,
            movie_idx_map,
        )

    @classmethod
    def __construct_user_idx_map(cls, record: MovieRating, acc: (dict, int)):
        user_idx_map, next_user_id = acc
        if user_idx_map.get(record.user_id) is None:
            user_idx_map[record.user_id] = next_user_id
            next_user_id += 1
        return user_idx_map, next_user_id

    @classmethod
    def __construct_movie_idx_map(cls, record: MovieRating, acc: (dict, int)):
        movie_idx_map, next_movie_id = acc
        if movie_idx_map.get(record.movie_id) is None:
            movie_idx_map[record.movie_id] = next_movie_id
            next_movie_id += 1
        return movie_idx_map, next_movie_id

    @classmethod
    def __construct_rating_map(cls, record: MovieRating, acc: dict):
        if acc.get(record.user_id) is None:
            acc[record.user_id] = {}
        acc[record.user_id][record.movie_id] = record
        return acc

    @classmethod
    def __calc_total_mean(cls, data: np.ndarray):
        return np.mean(data[np.isfinite(data)])

    @classmethod
    def __calc_mean_along_axis(cls, data: np.ndarray, axis: int) -> np.ndarray:
        return np.apply_along_axis(lambda x: np.mean(x[np.isfinite(x)]), axis, data)

    @classmethod
    def __fill_matrix_with_mean(cls, matrix: np.ndarray, mean: int):
        matrix[np.isinf(matrix)] = mean

    @classmethod
    def __fill_matrix_with_mean_by_axis(
        cls, matrix: np.ndarray, mean_by_axis: np.ndarray, axis: int
    ):
        for i in range(matrix.shape[axis]):
            if axis == 0:
                matrix[i][np.isinf(matrix[i])] = mean_by_axis[i]
            elif axis == 1:
                matrix[:, i][np.isinf(matrix[:, i])] = mean_by_axis[i]

    @classmethod
    def __fill_matrix(
        cls,
        matrix: np.ndarray,
        training_data: dict,
        user_idx_map: dict,
        movie_idx_map: dict,
    ):
        for user_id, data_in_user in training_data.items():
            for movie_id, record in data_in_user.items():
                user_idx = user_idx_map[user_id]
                movie_idx = movie_idx_map[movie_id]
                matrix[user_idx][movie_idx] = record.rating
