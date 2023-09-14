import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class AnalizaComponentePrincipale():
    def __init__(self, tabel, nume_variabile=None):
        if nume_variabile is None:
            nume_variabile = list(tabel.columns)

        self._x = tabel[nume_variabile].values

    def creare_model(self, standardizare=True, nlib=0):
        if standardizare:
            x_ = (self._x - np.mean(self._x, axis=0)) / np.std(self._x, axis=0)
        else:
            x_ = self._x - np.mean(self._x, axis=0)

        n, m = np.shape(self._x)

        r_cov = (1 / (n - nlib)) * x_.T @ x_

        valp, vectp = np.linalg.eig(r_cov)

        indici = np.flipud(np.argsort(valp))

        print("valp", valp)
        print("valp[indici]", valp[indici])

        self._alpha = valp[indici]
        self._a = vectp[:, indici]

        self._c = x_ @ self._a
        self.etichete_componente = ["Comp" + str(i) for i in range(m)]

        predicat = np.where(self._alpha > 1)
        print("Predicat Kaiser", predicat)
        print('Type predicat', type(predicat))
        self.nr_comp_kaiser = len(predicat[0])
        print("Nr comp Kaiser", self.nr_comp_kaiser)

        pondere = np.cumsum(self._alpha / sum(self._alpha))
        predicat = np.where(pondere > 0.8)
        print("Predicat procent", predicat)
        print('Type predicat', type(predicat))
        self.nr_comp_procent = predicat[0][0] + 1
        print("Nr comp procent", self.nr_comp_procent)

        eps = self._alpha[:(m - 1)] - self._alpha[1:]
        sigma = eps[:(m - 2)] - eps[1:]

        predicat = np.where(sigma < 0)
        print("Predicat Cattell", predicat)
        print('Type predicat', type(predicat))
        self.nr_comp_cattell = predicat[0][0] + 1
        print("Nr comp Cattell", self.nr_comp_cattell)

        self.r_x_c = np.corrcoef(self._x, self._c, rowvar=False)[:m, :m]

    def tabelare_varianta(self):
        procent = self.alpha * 100 / sum(self.alpha)
        tabel_varianta = pd.DataFrame(data={
            "Varianta": self.alpha,
            "Varianta cumulata": np.cumsum(self.alpha),
            "Procent varianta": procent,
            "Procent cumulat": np.cumsum(procent)
        },
            index=self.etichete_componente)

        return tabel_varianta

    def plot_varianta(self):
        fig = plt.figure("Plot varianta componente", figsize=(9, 9))
        ax = fig.add_subplot(1, 1, 1)

        ax.set_xlabel("Componente", fontdict={"fontsize": 12, "color": "b"})
        ax.set_ylabel("Varianta", fontdict={"fontsize": 12, "color": "b"})

        m = len(self.alpha)
        gradatii = np.arange(1, m+1)
        ax.set_xticks(gradatii)

        ax.plot(gradatii, self.alpha)
        ax.scatter(gradatii, self.alpha, color="r")

        ax.axhline(1, c="g", label="Kaiser")
        ax.axhline(self.alpha[self.nr_comp_procent - 1], c="m", label="Procent acoperire")
        ax.axhline(self.alpha[self.nr_comp_cattell - 1], c="c", label="Cattell")

        ax.legend()

    @property
    def x(self):
        return self._x

    @property
    def alpha(self):
        return self._alpha

    @property
    def a(self):
        return self._a

    @property
    def c(self):
        return self._c
