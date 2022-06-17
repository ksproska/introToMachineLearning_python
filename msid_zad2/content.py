# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 2: k-NN i Naive Bayes
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba, P. Dąbrowski
#  2019
# --------------------------------------------------------------------------

import numpy as np


class DistanceTuple:
    def __init__(self, dist, label):
        self.distance = dist
        self.label = label

    def lab(self):
        return self.label

    def __lt__(self, other):
        return self.distance < other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __le__(self, other):
        return self.distance <= other.distance

    def __ge__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance


def get_labels_tab(tab):
    labels = []
    for e in tab:
        if e not in labels:
            labels.append(e)
    return labels


def get_labels_size(matrix):
    return max([len(get_labels_tab(r)) for r in matrix])


def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    X = X.toarray()
    X_train = X_train.toarray()
    X_train_T = X_train.transpose()
    return (~X).astype(int) @ X_train_T + X.astype(int) @ (~X_train_T)


def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    N2 = len(y)
    N1 = Dist.shape[0]
    tuples = np.full(Dist.shape, None)
    for r in range(N1):
        for c in range(N2):
            tuples[r, c] = DistanceTuple(Dist[r, c], y[c])

        tuples[r] = np.sort(tuples[r], kind='mergesort')

        for c in range(N2):
            tuples[r, c] = tuples[r, c].lab()

    return tuples


def p_y_x_knn(y, k): # MLE
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """
    N1 = y.shape[0]
    M = get_labels_size(y)
    destination = np.full((N1, M), 0.0)
    n1 = 0
    for n1_row in y:
        for n2_k in range(k):
            label = n1_row[n2_k]
            destination[n1][label] += 1
        n1 += 1
    destination /= k
    return destination


def classification_error(p_y_x, y_true):
    # wybrac max wartosc, jak rowne to ostatnia
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """
    prediction = []
    for r in p_y_x:
        mx = None
        inx_mx = None
        inx = 0
        for e in r:
            if mx is None or e >= mx:
                mx = e
                inx_mx = inx
            inx += 1
        prediction.append(inx_mx)
    len_y = len(y_true)
    ret = sum([y_true[i] != prediction[i] for i in range(len_y)])/len_y
    # prediction = []
    # for r in p_y_x:
    #     mx = max(r.tolist())
    #     i_rev = r.tolist()[::-1].index(mx)
    #     prediction.append(len(r) - i_rev - 1)
    # ret = sum([y_true[i] != prediction[i] for i in range(len(y_true))]) / len(y_true)
    return ret


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    hd_X = hamming_distance(X_val, X_train)
    labels_sorted = sort_train_labels_knn(hd_X, y_train)
    errors = []
    for k in k_values:
        MLE = p_y_x_knn(labels_sorted, k)
        err = classification_error(MLE, y_val)
        errors.append(err)
    best_error = min(errors)
    best_k = k_values[errors.index(best_error)]
    return best_error, best_k, errors


def estimate_a_priori_nb(y_train):
    """
    Wyznacz rozkład a priori p(y) każdej z klas dla obiektów ze zbioru
    treningowego.

    :param y_train: etykiety dla danych treningowych 1xN
    :return: wektor prawdopodobieństw a priori p(y) 1xM
    """

    M = len(get_labels_tab(y_train))
    desin = []
    for i in range(M):
        desin.append(sum([x == i for x in y_train])/len(y_train))
    return desin


def estimate_p_x_y_nb(X_train, y_train, a, b):
    """
    Wyznacz rozkład prawdopodobieństwa p(x|y) zakładając, że *x* przyjmuje
    wartości binarne i że elementy *x* są od siebie niezależne.

    :param X_train: dane treningowe NxD
    :param y_train: etykiety klas dla danych treningowych 1xN
    :param a: parametr "a" rozkładu Beta
    :param b: parametr "b" rozkładu Beta
    :return: macierz prawdopodobieństw p(x|y) dla obiektów z "X_train" MxD.
    """
    M = len(get_labels_tab(y_train))
    D = X_train.shape[1]
    N = X_train.shape[0]
    destin = np.full((M, D), None)
    for d in range(D):
        for k in range(M):
            s1 = sum([y_train[n] == k and X_train[n, d] for n in range(N)]) + a - 1
            s2 = sum([y_train[n] == k for n in range(N)]) + a + b - 2
            destin[k, d] = s1/s2
    return destin


def p_y_x_nb(p_y, p_x_1_y, X): # MAP
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) dla każdej z klas z wykorzystaniem
    klasyfikatora Naiwnego Bayesa.

    :param p_y: wektor prawdopodobieństw a priori 1xM
    :param p_x_1_y: rozkład prawdopodobieństw p(x=1|y) MxD
    :param X: dane dla których beda wyznaczone prawdopodobieństwa, macierz NxD
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" NxM
    """
    M = p_x_1_y.shape[0]
    D = X.shape[1]
    N = X.shape[0]
    destin = np.full((N, M), None)
    for n in range(N):
        for m in range(M):
            numerator = 1.0
            for d in range(D):
                if X[n, d]:
                    numerator *= p_x_1_y[m, d]
                else:
                    numerator *= 1 - p_x_1_y[m, d]
            numerator *= p_y[m]
            destin[n, m] = numerator
        denominator = sum(destin[n])
        for m in range(M):
            destin[n, m] /= denominator

    return destin


def model_selection_nb(X_train, X_val, y_train, y_val, a_values, b_values):
    """
    Wylicz bład dla różnych wartości *a* i *b*. Dokonaj selekcji modelu Naiwnego
    Byesa, wyznaczając najlepszą parę wartości *a* i *b*, tj. taką, dla której
    wartość błędu jest najniższa.
    
    :param X_train: zbiór danych treningowych N2xD
    :param X_val: zbiór danych walidacyjnych N1xD
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param a_values: lista parametrów "a" do sprawdzenia
    :param b_values: lista parametrów "b" do sprawdzenia
    :return: krotka (best_error, best_a, best_b, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_a" i "best_b" to para parametrów
        "a" i "b" dla której błąd był najniższy, a "errors" - lista wartości
        błędów dla wszystkich kombinacji wartości "a" i "b" (w kolejności
        iterowania najpierw po "a_values" [pętla zewnętrzna], a następnie
        "b_values" [pętla wewnętrzna]).
    """
    errors = []
    pi = estimate_a_priori_nb(y_train)
    for a in a_values:
        errors_inner = []
        for b in b_values:
            teta = estimate_p_x_y_nb(X_train, y_train, a, b)
            MAP = p_y_x_nb(pi, teta, X_val)
            err = classification_error(MAP, y_val)
            errors_inner.append(err)
        errors.append(errors_inner)

    best_error = None
    best_a, best_b = None, None
    for a in range(len(a_values)):
        for b in range(len(b_values)):
            if best_error is None or errors[a][b] < best_error:
                best_error = errors[a][b]
                best_a = a_values[a]
                best_b = b_values[b]

    return best_error, best_a, best_b, errors
