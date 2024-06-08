import argparse
import time
import numpy as np
from movie_rating import MovieRating, RatingMatrix
from gradient_descent import sgd
from svd import truncated_svd, full_svd
from error_equations import (
    BiasedErrorEquation,
    BasicErrorEquation,
    BiasedPPErrorEquation,
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


def predict_rating(
    record: MovieRating,
    user_idx_map: dict,
    movie_idx_map: dict,
    prediction_matrix: np.ndarray,
    total_mean: int,
    mean_by_user: np.ndarray,
    mean_by_movie: np.ndarray,
):
    user_idx = user_idx_map.get(record.user_id)
    movie_idx = movie_idx_map.get(record.movie_id)
    if user_idx is not None and movie_idx is not None:
        return prediction_matrix[user_idx][movie_idx]
    if user_idx is not None and mean_by_user[user_idx] != np.Inf:
        return mean_by_user[user_idx]
    if movie_idx is not None and mean_by_movie[movie_idx] != np.Inf:
        return mean_by_movie[movie_idx]
    return total_mean


# parse command line arguments
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "training_data", metavar="FILE_PATH", type=str, help="a training data file path"
)
parser.add_argument(
    "test_data", metavar="FILE_PATH", type=str, help="a test data file path"
)
parser.add_argument(
    "--method",
    type=str,
    default="biased_sgd",
    help=(
        "the method to use (f_svd, truc_svd, sgd, biased_sgd, biased_sgd++) (default:"
        " biased_sgd)"
    ),
)

parser.add_argument(
    "--k",
    type=int,
    default=100,
    help="the number of latent features (default: 100)",
)

parser.add_argument(
    "--n_epochs",
    type=int,
    default=30,
    help="the number of epochs (default: 50)",
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.005,
    help="the learning rate (default: 0.005)",
)

parser.add_argument(
    "--regularization_rate",
    type=float,
    default=0.02,
    help="the regularization rate (default: 0.01)",
)
args = parser.parse_args()


# read data
training_data_path: str = args.training_data
test_data_path: str = args.test_data

print(f"read training_data ({training_data_path})")

rating_matrix = RatingMatrix(read_movie_data(training_data_path))
print(f"generate rating matrix: {rating_matrix.matrix.shape}")

# make recommendation matrix
print("")

start = time.time()

if args.method == "f_svd":
    filled_matrix = rating_matrix.get_filled_matrix()
    recommend_matrix = full_svd(filled_matrix)
elif args.method == "truc_svd":
    filled_matrix = rating_matrix.get_filled_matrix()
    recommend_matrix = truncated_svd(filled_matrix, args.k)
elif args.method == "sgd":
    recommend_matrix = sgd(
        BasicErrorEquation(),
        rating_matrix,
        args.k,
        args.regularization_rate,
        args.n_epochs,
        args.learning_rate,
    )
elif args.method == "biased_sgd":
    recommend_matrix = sgd(
        BiasedErrorEquation(),
        rating_matrix,
        args.k,
        args.regularization_rate,
        args.n_epochs,
        args.learning_rate,
    )
elif args.method == "biased_sgd++":
    recommend_matrix = sgd(
        BiasedPPErrorEquation(),
        rating_matrix,
        args.k,
        args.regularization_rate,
        args.n_epochs,
        args.learning_rate,
    )

end = time.time()
print(f"model training time: {end - start:.2f}s")

# test with test data
print("")
print("read test_data (", test_data_path, ")")

test_data_cnt = 0
d_sq_sum = 0
for record in read_movie_data(test_data_path):
    real_rating = record.rating
    predicted_rating = predict_rating(
        record,
        rating_matrix.user_idx_map,
        rating_matrix.movie_idx_map,
        recommend_matrix,
        rating_matrix.total_mean,
        rating_matrix.user_mean,
        rating_matrix.movie_mean,
    )

    d_sq_sum += (real_rating - predicted_rating) ** 2
    test_data_cnt += 1

print(f"RMSE: {np.sqrt(d_sq_sum / test_data_cnt):.4f}")
