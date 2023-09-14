import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hclust
from util import *
from acp import AnalizaComponentePrincipale as acp


def execute():
    tabel = pd.read_csv("date_tari.csv", index_col=0)
    variabile = list(tabel)[1:]
    instante = list(tabel.index)

    n = len(instante)
    m = len(variabile)

    x = tabel[variabile].values
    nan_replace(x)

    metoda = "ward"
    h = hclust.linkage(x, method=metoda)
    print("h", h)

    p = n - 1

    k_dif_max = np.argmax(h[1:, 2] - h[:(p - 1), 2])
    print("k_dif_max", k_dif_max)
    nr_clusteri = p - k_dif_max
    print(nr_clusteri)

    partitie_opt_3 = partitie(h, nr_clusteri, p, instante, metoda)
    print("partitie_opt_3", partitie_opt_3)

    partitie_opt_3_t = pd.DataFrame(
        data={"Cluster": partitie_opt_3},
        index=instante
    )
    partitie_opt_3_t.to_csv("partitie_opt_3.csv")

    partitie_opt_4 = partitie(h, 4, p, instante, metoda)
    print("partitie_opt_4", partitie_opt_4)

    partitie_opt_4_t = pd.DataFrame(
        data={"Cluster": partitie_opt_4},
        index=instante
    )
    partitie_opt_4_t.to_csv("partitie_opt_4.csv")

    partitie_opt_5 = partitie(h, 5, p, instante, metoda)
    print("partitie_opt_5", partitie_opt_5)

    partitie_opt_5_t = pd.DataFrame(
        data={"Cluster": partitie_opt_5},
        index=instante
    )
    partitie_opt_5_t.to_csv("partitie_opt_5.csv")

    for i in range(m):
        histograma(x[:,i], variabile[i],partitie_opt_3)

    # acp
    print("####################################################")
    nan_replace_a(tabel)

    nume_variabile = list(tabel.columns)[:]

    model = acp(tabel, nume_variabile)
    model.creare_model()

    tabel_varianta = model.tabelare_varianta()
    tabel_varianta.to_csv("Varianta.csv")

    model.plot_varianta()

    r_x_c = model.r_x_c
    tabel_rxc = tabelare_matrice(r_x_c, nume_variabile, model.etichete_componente, "Corelatii_factoriale.csv")

    corelograma(tabel_rxc)

    plot_corelatii(tabel_rxc, "Comp2", "Comp3")
    # plot_corelatii(tabel_rxc, "Comp1", "Comp3")

    tabel_componente = tabelare_matrice(model.c, tabel.index, model.etichete_componente, "Componente.csv")
    plot_componente(tabel_componente, "Comp1", "Comp2")

    componente_patrat = model.c * model.c
    cosin = np.transpose(componente_patrat.T / np.sum(componente_patrat, axis=1))
    _ = tabelare_matrice(cosin, tabel.index, model.etichete_componente, "Cosin.csv")

    comunalitati = np.cumsum(r_x_c * r_x_c, axis=1)
    tabel_comunalitati = tabelare_matrice(comunalitati, nume_variabile, model.etichete_componente, "Comunalitati.csv")

    corelograma(tabel_comunalitati, valMin=0, titlu="Comunalitati")

    show()


if __name__ == "__main__":
    execute()
