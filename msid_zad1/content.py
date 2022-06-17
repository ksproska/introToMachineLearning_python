# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
import sys

import numpy as np

from utils import polynomial


def mean_squared_error(x, y, w):
    """
        :param x: ciąg wejściowy Nx1
        :param y: ciąg wyjsciowy Nx1
        :param w: parametry modelu (M+1)x1
        :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
         uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    n = x.shape[0]
    poly = (y - polynomial(x, w))**2
    return (1.0/n)*np.sum(poly)


def design_matrix(x_train, M):
    """
        :param x_train: ciąg treningowy Nx1
        :param M: stopień wielomianu 0,1,2,...
        :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """
    a = np.full((x_train.shape[0], M+1), 0.0)
    for i in range(x_train.shape[0]):
        for j in range(M+1):
            a[i][j] = x_train[i][0]**j
    return a


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    q = design_matrix(x_train, M)
    qT = q.transpose()
    w = np.linalg.inv(qT @ q) @ qT @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return w, err


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    q = design_matrix(x_train, M)
    qT = q.transpose()
    w = np.linalg.inv(qT @ q + np.identity(M+1)*regularization_lambda) @ qT @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return w, err


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """
    k = (None, None, None)
    for M in M_values:
        w_err_train = least_squares(x_train, y_train, M)
        w = w_err_train[0]
        err_train = w_err_train[1]
        err_val = mean_squared_error(x_val, y_val, w)
        if k[0] is None or k[1] + k[2] > err_train + err_val:
            k = (w, err_train, err_val)
    return k


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    k = (None, None, None, None)
    for L in lambda_values:
        w_err_train = regularized_least_squares(x_train, y_train, M, L)
        w = w_err_train[0]
        err_train = w_err_train[1]
        err_val = mean_squared_error(x_val, y_val, w)
        if k[0] is None or k[1] + k[2] > err_train + err_val:
            k = (w, err_train, err_val, L)
    return k
