import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sb
import scipy.cluster.hierarchy as hclust


def nan_replace(x):
    is_nan = np.isnan(x)
    k_nan = np.where(is_nan)
    x[k_nan] = np.nanmean(x[:, k_nan[1]], axis=0)


def partitie(h, nr_clusteri, p, instante, metoda):
    k_div_max = p - nr_clusteri
    prag = (h[k_div_max, 2] + h[k_div_max + 1, 2]) / 2

    plot_ierarhie(h, instante, "Partitia cu " + str(nr_clusteri) + " clusteri. Metoda: " + metoda, prag)

    n = p + 1

    c = np.arange(n)
    for i in range(n - nr_clusteri):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k1] = n + i
        c[c == k2] = n + i

    coduri = pd.Categorical(c).codes
    return np.array(["c" + str(cod + 1) for cod in coduri])


def plot_ierarhie(h, instante, titlu, prag=None):
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontsize=18, color='b')
    hclust.dendrogram(h, labels=instante, ax=ax, color_threshold=prag)


def histograma(x, variabila, partitia):
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle("Histograme pt variabila " + variabila, fontsize=18, color='b')

    clusteri = list(set(partitia))
    size = len(clusteri)

    axs = fig.subplots(1, size, sharey=True)
    for i in range(size):
        ax = axs[i]
        ax.set_xlabel(clusteri[i])
        ax.hist(x[partitia == clusteri[i]], bins=10, rwidth=0.9, range=(min(x), max(x)))


def nan_replace_a(t):
    assert isinstance(t, pd.DataFrame)
    nume_variabile = list(t.columns)

    for each in nume_variabile:
        if any(t[each].isna()):
            if pd.api.types.is_numeric_dtype(t[each]):
                t[each].fillna(t[each].mean(), inplace=True)
            else:
                modul = t[each].mode()[0]
                t[each].fillna(modul, inplace=True)


def tabelare_matrice(x, nume_linii=None, nume_coloane=None, out=None):
    t = pd.DataFrame(x, nume_linii, nume_coloane)
    if out is not None:
        t.to_csv(out)

    return t


def corelograma(x, valMin=-1, valMax=1, titlu="Corelatii factoriale"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})

    ax_ = sb.heatmap(data=x,
               vmin=valMin, vmax=valMax,
               cmap='RdYlBu', annot=True, ax=ax)
    ax_.set_xticklabels(x.columns, ha="right", rotation=30)


def plot_corelatii(x, var_x, var_y, titlu="Plot corelatii", aspect="auto"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel(var_x, fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel(var_y, fontdict={"fontsize": 12, "color": "b"})

    ax.set_aspect(aspect)

    theta = np.arange(0, 2 * np.pi, 0.01)
    ax.plot(np.cos(theta), np.sin(theta), color="b")

    ax.axhline(0)
    ax.axvline(0)

    ax.scatter(x[var_x], x[var_y], color="r")

    for i in range(len(x)):
        ax.text(x[var_x].iloc[i], x[var_y].iloc[i], x.index[i])


def plot_componente(x, var_x, var_y, titlu="Plot componente", aspect="auto"):
    fig = plt.figure(titlu, figsize=(9, 9))
    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel(var_x, fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel(var_y, fontdict={"fontsize": 12, "color": "b"})

    ax.set_aspect(aspect)
    ax.scatter(x[var_x], x[var_y], color="r")

    for i in range(len(x)):
        ax.text(x[var_x].iloc[i], x[var_y].iloc[i], x.index[i])


def show():
    plt.show()
