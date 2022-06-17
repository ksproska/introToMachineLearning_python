# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 3: Regresja logistyczna
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


# funkcja sigmoidalna
def sigmoid(x):
    """
    Wylicz wartość funkcji sigmoidalnej dla punktów *x*.

    :param x: wektor wartości *x* do zaaplikowania funkcji sigmoidalnej Nx1
    :return: wektor wartości funkcji sigmoidalnej dla wartości *x* Nx1
    """
    return 1/(1+np.exp(-x))


# funkcja celu
def logistic_cost_function(w, x_train, y_train):
    """
    Wylicz wartość funkcji logistycznej oraz jej gradient po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej,
        a *grad* jej gradient po parametrach *w* Mx1
    """
    N = x_train.shape[0]
    wT, x_trainT = w.transpose(), x_train.transpose()
    log_sum = 0.0

    for n in range(N):
        sig_pos = sigmoid(wT @ x_train[n])
        yn = y_train[n]
        log_sum += yn*np.log(sig_pos) + (1 - yn)*np.log(1 - sig_pos)

    log_val = - log_sum/N
    grad = 1/N * x_trainT @ (sigmoid(x_train @ w) - y_train)
    return log_val, grad


# algorytm gradientu prostego
def gradient_descent(obj_fun, w0, epochs, eta):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą algorytmu gradientu
    prostego, korzystając z kroku uczenia *eta* i zaczynając od parametrów *w0*.
    Wylicz wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość
    parametrów modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argument
        wektor parametrów *w* [wywołanie *val, grad = obj_fun(w)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok algorytmu gradientu prostego
    :param eta: krok uczenia
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu w każdej
        epoce (lista o długości *epochs*)
    """
    log_values = []
    w = w0
    _, grad = obj_fun(w)
    for k in range(epochs):
        w = w - eta*grad
        val, grad = obj_fun(w)
        log_values.append(val[0])
    return w, log_values


# algorytm stochastycznego gradientu prostego
def stochastic_gradient_descent(obj_fun, x_train, y_train, w0, epochs, eta, mini_batch):
    """
    Dokonaj *epochs* aktualizacji parametrów modelu metodą stochastycznego
    algorytmu gradientu prostego, korzystając z kroku uczenia *eta*, paczek
    danych o rozmiarze *mini_batch* i zaczynając od parametrów *w0*. Wylicz
    wartość funkcji celu *obj_fun* w każdej iteracji. Wyznacz wartość parametrów
    modelu w ostatniej epoce.

    :param obj_fun: optymalizowana funkcja celu, przyjmująca jako argumenty
        wektor parametrów *w*, paczkę danych składających się z danych
        treningowych *x* i odpowiadających im etykiet *y*
        [wywołanie *val, grad = obj_fun(w, x, y)*]
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :return: krotka (w, log_values), gdzie *w* to znaleziony optymalny
        punkt *w*, a *log_values* to lista wartości funkcji celu dla całego
        zbioru treningowego w każdej epoce (lista o długości *epochs*)
    """
    M = x_train.shape[0]//mini_batch
    x_batches, y_batches = [], []
    for m in range(M):
        start = m * mini_batch
        stop = start + mini_batch
        x_batches.append(x_train[start:stop, :])
        y_batches.append(y_train[start:stop, :])

    log_values = []
    w = w0
    for k in range(epochs):
        for m in range(M):
            _, grad = obj_fun(w, x_batches[m], y_batches[m])
            w = w - eta * grad

        log_values.append(obj_fun(w, x_train, y_train)[0][0])
    return w, log_values


# funkcja celu z regularyzacja
def regularized_logistic_cost_function(w, x_train, y_train, regularization_lambda):
    """
    Wylicz wartość funkcji logistycznej z regularyzacją l2 oraz jej gradient
    po parametrach.

    :param w: wektor parametrów modelu Mx1
    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param regularization_lambda: parametr regularyzacji l2
    :return: krotka (log, grad), gdzie *log* to wartość funkcji logistycznej
        z regularyzacją l2, a *grad* jej gradient po parametrach *w* Mx1
    """
    N = x_train.shape[0]
    wT, x_trainT = w.transpose(), x_train.transpose()
    log_sum = 0.0

    for n in range(N):
        sig_pos = sigmoid(wT @ x_train[n])
        yn = y_train[n]
        log_sum += yn * np.log(sig_pos) + (1 - yn) * np.log(1 - sig_pos)

    regularization_diff_val = (regularization_lambda / 2) * sum((w[1:] ** 2))
    log_val = (- log_sum / N) + regularization_diff_val

    regularization_diff_matrix = regularization_lambda * w
    regularization_diff_matrix[0][0] = 0
    grad = 1/N * x_trainT @ (sigmoid(x_train@w) - y_train) + regularization_diff_matrix

    return log_val, grad


# regresja logistyczna
def prediction(x, w, theta):
    """
    Wylicz wartości predykowanych etykiet dla obserwacji *x*, korzystając
    z modelu o parametrach *w* i progu klasyfikacji *theta*.

    :param x: macierz obserwacji NxM
    :param w: wektor parametrów modelu Mx1
    :param theta: próg klasyfikacji z przedziału [0,1]
    :return: wektor predykowanych etykiet ze zbioru {0, 1} Nx1
    """
    N = x.shape[0]
    y_pred = np.full((N, 1), 0)
    wT = w.transpose()

    for n in range(N):
        sig_pos = sigmoid(wT @ x[n])
        y_pred[n, 0] = sig_pos >= theta
    return y_pred


# miara f-measure
def f_measure(y_true, y_pred):
    """
    Wylicz wartość miary F (F-measure) dla zadanych rzeczywistych etykiet
    *y_true* i odpowiadających im predykowanych etykiet *y_pred*.

    :param y_true: wektor rzeczywistych etykiet Nx1
    :param y_pred: wektor etykiet predykowanych przed model Nx1
    :return: wartość miary F (F-measure)
    """
    N = y_true.shape[0]
    TP, FP, FN = 0, 0, 0
    for n in range(N):
        real_val, pred_val = y_true[n], y_pred[n]
        if real_val & pred_val:
            TP += 1
        if (not real_val) & pred_val:
            FP += 1
        if real_val & (not pred_val):
            FN += 1
    return 2*TP/(2*TP + FP + FN)


def model_selection(x_train, y_train, x_val, y_val, w0, epochs, eta, mini_batch, lambdas, thetas):
    """
    Policz wartość miary F dla wszystkich kombinacji wartości regularyzacji
    *lambda* i progu klasyfikacji *theta. Wyznacz parametry *w* dla modelu
    z regularyzacją l2, który najlepiej generalizuje dane, tj. daje najmniejszy
    błąd na ciągu walidacyjnym.

    :param x_train: zbiór danych treningowych NxM
    :param y_train: etykiety klas dla danych treningowych Nx1
    :param x_val: zbiór danych walidacyjnych NxM
    :param y_val: etykiety klas dla danych walidacyjnych Nx1
    :param w0: początkowy wektor parametrów *w* Mx1
    :param epochs: liczba epok stochastycznego algorytmu gradientu prostego
    :param eta: krok uczenia
    :param mini_batch: rozmiar paczki danych / mini-batcha
    :param lambdas: lista wartości parametru regularyzacji l2 *lambda*,
        które mają być sprawdzone
    :param thetas: lista wartości progów klasyfikacji *theta*,
        które mają być sprawdzone
    :return: krotka (regularization_lambda, theta, w, F), gdzie
        *regularization_lambda* to wartość regularyzacji *lambda* dla
        najlepszego modelu, *theta* to najlepszy próg klasyfikacji,
        *w* to parametry najlepszego modelu, a *F* to macierz wartości miary F
        dla wszystkich par *(lambda, theta)* #lambda x #theta
    """
    w, f_max = None, None
    lambda_f_max, theta_f_max = None, None
    lambdas_len, thetas_len = len(lambdas), len(thetas)
    F = np.full((lambdas_len, thetas_len), 0.0)

    for l in range(lambdas_len):
        lambda_val = lambdas[l]
        w, _ = stochastic_gradient_descent(
            lambda x, y, z: regularized_logistic_cost_function(x, y, z, lambda_val),
            x_train, y_train, w0, epochs, eta, mini_batch)

        for t in range(thetas_len):
            theta_val = thetas[t]
            y_pred = prediction(x_val, w, theta_val)
            f_val = f_measure(y_val, y_pred)
            F[l][t] = f_val
            if f_max is None or f_max < f_val:
                f_max, lambda_f_max, theta_f_max = f_val, lambda_val, theta_val

    return lambda_f_max, theta_f_max, w, F
