import argparse
import numpy as np
from surprise import Dataset, Reader, SVD, SVDpp, accuracy
from pandas import DataFrame
import time


def parse_movie_data_to_array(data: str) -> np.ndarray:
    data = data.strip()
    data = data.split("\t")
    return np.array([data[0], data[1], float(data[2])])


def read_movie_data_to_array(filepath):
    file = open(filepath, "rt")
    while True:
        line = file.readline()
        if line == "":
            break
        yield parse_movie_data_to_array(line)
    file.close()


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
    default="biased_sgd++",
    help="the method to use (biased_sgd, biased_sgd++) (default: biased_sgd++)",
)

parser.add_argument(
    "--k",
    type=int,
    default=30,
    help="the number of latent features (default: 30)",
)

parser.add_argument(
    "--n_epochs",
    type=int,
    default=50,
    help="the number of epochs (default: 50)",
)

parser.add_argument(
    "--learning_rate",
    type=float,
    default=0.01,
    help="the learning rate (default: 0.01)",
)

parser.add_argument(
    "--regularization_rate",
    type=float,
    default=0.01,
    help="the regularization rate (default: 0.01)",
)
args = parser.parse_args()

# read data
training_data_path: str = args.training_data
test_data_path: str = args.test_data

print(f"read training_data ({training_data_path}), test_data ({test_data_path})")

train_arr = read_movie_data_to_array(training_data_path)
train_df = DataFrame(
    train_arr,
    columns=["user_id", "movie_id", "rating"],
)
train_df = train_df.astype(
    {"user_id": "int64", "movie_id": "int64", "rating": "float32"}
)

test_arr = read_movie_data_to_array(test_data_path)
test_df = DataFrame(
    test_arr,
    columns=["user_id", "movie_id", "rating"],
)
test_df = test_df.astype({"user_id": "int64", "movie_id": "int64", "rating": "float32"})

reader = Reader(rating_scale=(0.5, 5.0))

trainset = Dataset.load_from_df(
    train_df[["user_id", "movie_id", "rating"]], reader
).build_full_trainset()

train_mean = train_df["rating"].mean()

testset = (
    Dataset.load_from_df(test_df[["user_id", "movie_id", "rating"]], reader)
    .build_full_trainset()
    .build_testset()
)

# make recommendation matrix
print("")
print(
    f"method: {args.method}, "
    f"k: {args.k}, n_epochs: {args.n_epochs}, learning_rate: {args.learning_rate},"
    f" regularization_rate: {args.regularization_rate}"
)

if args.method == "sgd":
    algo = SVD(
        n_factors=args.k,
        n_epochs=args.n_epochs,
        lr_all=args.learning_rate,
        reg_all=args.regularization_rate,
        biased=False,
        init_mean=np.sqrt(train_mean / args.k),
    )
elif args.method == "biased_sgd":
    algo = SVD(
        n_factors=args.k,
        n_epochs=args.n_epochs,
        lr_all=args.learning_rate,
        reg_all=args.regularization_rate,
    )
elif args.method == "biased_sgd++":
    algo = SVDpp(
        n_factors=args.k,
        n_epochs=args.n_epochs,
        lr_all=args.learning_rate,
        reg_all=args.regularization_rate,
    )

start = time.time()
algo.fit(trainset)
end = time.time()
print(f"model training time: {end - start:.2f}s")

predictions = algo.test(testset)
accuracy.rmse(predictions)
