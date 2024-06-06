import numpy as np
import random


def stochastic_gradient_descent(
    matrix: np.ndarray,
    k: int,
    n_epochs: int,
    learning_rate: float,
    regularization_rate: float,
    total_mean: float,
):
    print("stochastic gradient descent")
    print(
        f"k: {k}, n_epochs: {n_epochs}, learning_rate: {learning_rate},"
        f" regularization_rate: {regularization_rate}"
    )
    # 전체 평균으로 초기화
    # init_val = np.sqrt(total_mean / k)
    # p = np.full((matrix.shape[0], k), init_val)
    # q = np.full((k, matrix.shape[1]), init_val)

    # 정규 분포를 이용한 초기화
    init_val = np.sqrt(total_mean / k)
    sigma = np.sqrt(1 / k)
    p = np.full((matrix.shape[0], k), random.normalvariate(init_val, sigma))
    q = np.full((k, matrix.shape[1]), random.normalvariate(init_val, sigma))

    # 균등 분포를 이용한 초기화
    # max_val = np.sqrt(5 / k)
    # p = np.full((matrix.shape[0], k), max_val * random.random())
    # q = np.full((k, matrix.shape[1]), max_val * random.random())

    train_error = 0
    for epoch in range(n_epochs):
        train_error = 0
        train_cnt = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isinf(matrix[i][j]):
                    continue
                error = __calc_error(matrix, p, q, i, j, regularization_rate)
                p[i] += learning_rate * (error * q[:, j] - regularization_rate * p[i])
                q[:, j] += learning_rate * (
                    error * p[i] - regularization_rate * q[:, j]
                )
                train_error += error**2
                train_cnt += 1
        train_error /= train_cnt
        print(f"epoch: {epoch}, train_error: {train_error:.4f}")
    print(f"train_error: {train_error:.4f}")
    return p @ q


def __calc_error(
    matrix: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    r: int,
    c: int,
    regularization_rate: float,
):
    return (
        matrix[r][c]
        - p[r] @ q[:, c]
        + regularization_rate * (np.linalg.norm(p[r], 2) + np.linalg.norm(q[:, c], 2))
    )
