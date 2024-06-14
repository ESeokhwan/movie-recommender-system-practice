from .error_equations import ErrorEquation
from .movie_rating import RatingMatrix


def sgd(
    error_eq: ErrorEquation,
    matrix: RatingMatrix,
    k: int,
    regularization_rate: float,
    n_epochs: int,
    learning_rate: float,
    verbose: bool = False,
):
    print("stochastic gradient descent")
    print(
        f"method: {error_eq.__class__.__name__}, "
        f"k: {k}, n_epochs: {n_epochs}, learning_rate: {learning_rate},"
        f" regularization_rate: {regularization_rate}"
    )

    error_eq.init_params(matrix.matrix, matrix.total_mean, k, regularization_rate)

    train_error = 0
    for epoch in range(n_epochs):
        if not error_eq.update_params(learning_rate):
            print("Unexpected Termination -- Error in updating params")
            break
        if verbose is True:
            train_error = error_eq.calc_error()
            print(f"epoch: {epoch}, train_error: {train_error:.4f}")

    if verbose is True:
        print(f"train_error: {train_error:.4f}")
    return error_eq.predict()
