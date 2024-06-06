import argparse
import numpy as np


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


def parse_movie_data(data: str) -> MovieRating:
    data = data.strip()
    data = data.split("\t")
    return MovieRating(data[0], data[1], float(data[2]), int(data[3]))


def read_movie_data(filepath):
    file = open(filepath, "rt")
    while True:
        line = file.readline()
        if line == "":
            break
        yield parse_movie_data(line)
    file.close()


def __construct_user_idx_map(record: MovieRating, acc: (dict, int)):
    user_idx_map, next_user_id = acc
    if user_idx_map.get(record.user_id) is None:
        user_idx_map[record.user_id] = next_user_id
        next_user_id += 1
    return user_idx_map, next_user_id


def __construct_movie_idx_map(record: MovieRating, acc: (dict, int)):
    movie_idx_map, next_movie_id = acc
    if movie_idx_map.get(record.movie_id) is None:
        movie_idx_map[record.movie_id] = next_movie_id
        next_movie_id += 1
    return movie_idx_map, next_movie_id


def __construct_rating_map(record: MovieRating, acc: dict):
    if acc.get(record.user_id) is None:
        acc[record.user_id] = {}
    acc[record.user_id][record.movie_id] = record
    return acc


def __construct_mean_rating(record: MovieRating, acc: (int, int)):
    return acc[0] + record.rating, acc[1] + 1


def __construct_user_mean_rating(record: MovieRating, acc: dict):
    if acc.get(record.user_id) is None:
        acc[record.user_id] = (0, 0)
    acc[record.user_id] = (
        acc[record.user_id][0] + record.rating,
        acc[record.user_id][1] + 1,
    )
    return acc


def __construct_movie_mean_rating(record: MovieRating, acc: dict):
    if acc.get(record.movie_id) is None:
        acc[record.movie_id] = (0, 0)
    acc[record.movie_id] = (
        acc[record.movie_id][0] + record.rating,
        acc[record.movie_id][1] + 1,
    )
    return acc


def __init_rating_matrix_with_mean(rating_matrix: np.ndarray, mean: int):
    for i in range(rating_matrix.shape[0]):
        for j in range(rating_matrix.shape[1]):
            rating_matrix[i][j] = mean


def __init_rating_matrix_with_mean_by_user(
    rating_matrix: np.ndarray, mean_by_user: dict, user_idx_map: dict
):
    for user_id, mean_i in mean_by_user.items():
        idx = user_idx_map[user_id]
        mean = mean_i[0] / mean_i[1]
        rating_matrix[idx][:] = mean


def __init_rating_matrix_with_mean_by_movie(
    rating_matrix: np.ndarray, mean_by_movie: dict, movie_idx_map: dict
):
    for movie_id, mean_i in mean_by_movie.items():
        idx = movie_idx_map[movie_id]
        mean = mean_i[0] / mean_i[1]
        for i in range(rating_matrix.shape[0]):
            rating_matrix[i][idx] = mean


def __fill_rating_matrix(
    rating_matrix: np.ndarray,
    training_data: dict,
    user_idx_map: dict,
    movie_idx_map: dict,
):
    for user_id, data_in_user in training_data.items():
        for movie_id, record in data_in_user.items():
            user_idx = user_idx_map[user_id]
            movie_idx = movie_idx_map[movie_id]
            rating_matrix[user_idx][movie_idx] = record.rating


def generate_rating_matrix(data: iter):
    training_data = {}

    user_idx_map_i = ({}, 0)
    movie_idx_map_i = ({}, 0)

    total_mean_i = (0, 0)
    mean_by_user = {}
    mean_by_movie = {}

    for record in data:
        user_idx_map_i = __construct_user_idx_map(record, user_idx_map_i)
        movie_idx_map_i = __construct_movie_idx_map(record, movie_idx_map_i)
        training_data = __construct_rating_map(record, training_data)
        total_mean_i = __construct_mean_rating(record, total_mean_i)
        mean_by_user = __construct_user_mean_rating(record, mean_by_user)
        mean_by_movie = __construct_movie_mean_rating(record, mean_by_movie)

    total_mean = total_mean_i[0] / total_mean_i[1]
    user_idx_map = user_idx_map_i[0]
    movie_idx_map = movie_idx_map_i[0]

    rating_matrix = np.zeros((user_idx_map.__len__(), movie_idx_map.__len__()))
    __init_rating_matrix_with_mean(rating_matrix, total_mean)
    __init_rating_matrix_with_mean_by_movie(rating_matrix, mean_by_movie, movie_idx_map)
    __init_rating_matrix_with_mean_by_user(rating_matrix, mean_by_user, user_idx_map)
    __fill_rating_matrix(rating_matrix, training_data, user_idx_map, movie_idx_map)

    return (
        rating_matrix,
        user_idx_map,
        movie_idx_map,
        total_mean,
        mean_by_user,
        mean_by_movie,
    )


# generate prediction matrix


# SVD
def full_svd(rating_matrix: np.ndarray) -> np.ndarray:
    U, s_diag, Vt = np.linalg.svd(rating_matrix)
    print("single value decomposition")
    print(f"U: {U.shape}, s: {s_diag.shape}, Vt: {Vt.shape}")

    s = np.zeros((U.shape[1], Vt.shape[0]))
    s[: s_diag.shape[0], : s_diag.shape[0]] = np.diag(s_diag)

    return U @ s @ Vt


def truncated_svd(rating_matrix: np.ndarray, k: int) -> np.ndarray:
    U, s_diag, Vt = np.linalg.svd(rating_matrix)
    print("truncated single value decomposition")

    U = U[:, :k]
    s_diag = s_diag[:k]
    Vt = Vt[:k, :]
    print(f"U: {U.shape}, s: {s_diag.shape}, Vt: {Vt.shape}")

    s = np.zeros((U.shape[1], Vt.shape[0]))
    s[: s_diag.shape[0], : s_diag.shape[0]] = np.diag(s_diag)

    return U @ s @ Vt


def predict_rating(
    record: MovieRating,
    user_idx_map: dict,
    movie_idx_map: dict,
    prediction_matrix: np.ndarray,
    total_mean: int,
    mean_by_user: dict,
    mean_by_movie: dict,
):
    user_idx = user_idx_map.get(record.user_id)
    movie_idx = movie_idx_map.get(record.movie_id)
    if user_idx is not None and movie_idx is not None:
        return prediction_matrix[user_idx][movie_idx]
    if mean_by_user is not None and mean_by_user.get(record.user_id) is not None:
        return mean_by_user[record.user_id][0] / mean_by_user[record.user_id][1]
    if mean_by_movie is not None and mean_by_movie.get(record.movie_id) is not None:
        return mean_by_movie[record.movie_id][0] / mean_by_movie[record.movie_id][1]
    return total_mean


# parse command line arguments
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "training_data", metavar="FILE_PATH", type=str, help="a training data file path"
)
parser.add_argument(
    "test_data", metavar="FILE_PATH", type=str, help="a test data file path"
)
args = parser.parse_args()


# read data
training_data_path: str = args.training_data
test_data_path: str = args.test_data

print(f"read training_data ({training_data_path})")
rating_matrix, user_idx_map, movie_idx_map, total_mean, mean_by_user, mean_by_movie = (
    generate_rating_matrix(read_movie_data(training_data_path))
)

print(f"generate rating matrix: {rating_matrix.shape}")

# make recommendation matrix
print("")
# recommend_matrix = full_svd(rating_matrix)
recommend_matrix = truncated_svd(rating_matrix, 10)


# test with test data
print("")
print("read test_data (", test_data_path, ")")

test_data_cnt = 0
d_sq_sum = 0
for record in read_movie_data(test_data_path):
    real_rating = record.rating
    predicted_rating = predict_rating(
        record,
        user_idx_map,
        movie_idx_map,
        recommend_matrix,
        total_mean,
        mean_by_user,
        mean_by_movie,
    )

    d_sq_sum += (real_rating - predicted_rating) ** 2
    test_data_cnt += 1

print(f"RMSE: {np.sqrt(d_sq_sum / test_data_cnt)}")
