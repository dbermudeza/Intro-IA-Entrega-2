"""Genera el documento de diseño de la Práctica 2 en formato Word (.docx) a partir de lo que exporta el notebook en salidas/.

Uso:
    python build_informe.py [carpeta_salidas] [informe.docx] [--pdf]

* Todas las cifras salen de salidas/*.csv|json (los mismos archivos que escribe la última celda del notebook).
* El texto interpretativo se redactó con la corrida de referencia (cédula en salidas/resumen.json). Cada afirmación cualitativa
  se comprueba con `chequear()`: si tras cambiar la cédula alguna ya no se cumple, la consola imprime «REVISAR TEXTO».
* El contenido se define una sola vez como una lista de bloques (`B`) y se dibuja con `render_docx()`.
* `--pdf` convierte además el .docx a PDF con LibreOffice (opcional; también se puede exportar desde Word o Google Docs).
Requiere: pandas, numpy, matplotlib, python-docx.
"""
import json, os, re, shutil, subprocess, sys, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from docx import Document
from docx.enum.section import WD_ORIENT, WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

# ----------------------------------------------------------------------------------------------------------------------
# Parámetros
# ----------------------------------------------------------------------------------------------------------------------
args = [a for a in sys.argv[1:] if not a.startswith("--")]
HERE = os.path.dirname(os.path.abspath(__file__))


def _find_salidas():
    for p in ([args[0]] if args else []) + ["salidas", os.path.join(HERE, "salidas"), os.path.join(HERE, "..", "salidas"),
                                           os.path.join(HERE, "..", "..", "salidas")]:
        if os.path.exists(os.path.join(p, "resumen.json")):
            return os.path.abspath(p)
    sys.exit("No encuentro la carpeta salidas/ (debe contener resumen.json). Pásela como primer argumento.")


D = _find_salidas()
OUT_DOCX = os.path.abspath(args[1] if len(args) > 1 else "Informe_Practica2.docx")
WORK = tempfile.mkdtemp(prefix="informe_")           # diagramas generados aquí; no se ensucia salidas/

# ======================= EDITAR ANTES DE ENTREGAR =======================
INTEGRANTES = ["Juan José Jiménez Maya", "Thomas Tamayo Durango", "Daniel Bermudez Aranguren", "Juan David Giraldo"]
VIDEO_URL = ""        # pegar aquí el enlace de YouTube cuando el video esté subido; vacío = se deja un marcador resaltado
DATASET_URL = "https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india"
# ========================================================================

J = json.load(open(f"{D}/resumen.json", encoding="utf-8"))
R = pd.read_csv(f"{D}/resultados_supervisado.csv")
F3 = pd.read_csv(f"{D}/figura3_tabla.csv")
T3 = pd.read_csv(f"{D}/tres_datos_prueba_supervisado.csv")
T3u = pd.read_csv(f"{D}/tres_datos_prueba_no_supervisado.csv")
MODELOS = ["Árboles", "KNN", "SVM", "Redes Neuronales"]
CLASES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CASOS = {i + 1: dict(ED=(i >= 4), OUT=(i % 4 >= 2), BAL=(i % 2 == 1)) for i in range(8)}
SI = lambda b: "SI" if b else "NO"
fmt = lambda x, n=3: f"{x:.{n}f}"
sg = lambda x: "≈ 0" if abs(x) < 0.0005 else f"{x:+.3f}"
pct = lambda x, n=0: f"{100 * x:.{n}f} %"

P = {m: R.pivot(index="caso", columns="modelo", values=m)[MODELOS] for m in ["f1", "accuracy", "precision", "recall"]}
f1 = P["f1"]
for k, key in [("ED", "ED"), ("OUT", "OUT"), ("BAL", "BAL")]:
    R[k] = R.caso.map(lambda c, key=key: SI(CASOS[c][key]))


def efecto(factor):
    """F1 medio con el factor en SI y en NO, y su diferencia (SI - NO), por modelo."""
    t = R.groupby(["modelo", factor]).f1.mean().unstack().reindex(MODELOS)
    return t["NO"], t["SI"], t["SI"] - t["NO"]


ef = {k: efecto(k) for k in ["ED", "OUT", "BAL"]}
ED_, OUT_, BAL_ = (ef[k][2] for k in ["ED", "OUT", "BAL"])
roc = J["roc_caso8"]
casos_tr = J["casos"]
nul = J["nulos_pct"]
mx = {m: (f1[m].max(), int(f1[m].idxmax())) for m in MODELOS}
ranking = sorted(mx.items(), key=lambda kv: -kv[1][0])
c8 = f1.loc[8]
n_feat = casos_tr["1"]["n_features"]
n_train_unb, n_train_bal = casos_tr["1"]["n_train"], casos_tr["2"]["n_train"]
ultimo_dig = (J["n_registros"] - 5000) // 150
dist = J["dist_subconjunto"]
n_test = J["tamanos_split"]["te"]
arbol = J["arbol"]
knn_k = J["knn_k"]
nn_ep = J["nn_epocas"]
k_bal = [knn_k[str(c)] for c in range(1, 9) if CASOS[c]["BAL"]]
k_unb = [knn_k[str(c)] for c in range(1, 9) if not CASOS[c]["BAL"]]


def rep(m):
    return pd.read_csv(f"{D}/reporte_clases_caso8_{m.replace(' ', '_')}.csv", index_col=0)


def cmat(m):
    return pd.read_csv(f"{D}/matriz_confusion_caso8_{m.replace(' ', '_')}.csv", index_col=0)


sev = {m: rep(m).loc["Severe", "recall"] for m in MODELOS}
poor_f1 = {m: rep(m).loc["Poor", "f1-score"] for m in MODELOS}
tot_test = cmat("SVM").values.sum()
adj = {}
for m in MODELOS:
    cm = cmat(m).values
    err = cm.sum() - np.trace(cm)
    adj[m] = sum(cm[i, j] for i in range(6) for j in range(6) if abs(i - j) == 1) / err
mod_share = cmat("SVM").loc["Moderate"].sum() / tot_test      # proporción de Moderate en el test

AVISOS = []


def chequear(cond, msg):
    """Registra una advertencia si una afirmación del texto deja de cumplirse con los datos actuales."""
    if not cond:
        AVISOS.append(msg)
    return cond


# ----------------------------------------------------------------------------------------------------------------------
# Verificación de las afirmaciones cualitativas
# ----------------------------------------------------------------------------------------------------------------------
chequear(all(np.allclose(f1.loc[i, "Árboles"], f1.loc[i + 4, "Árboles"]) for i in range(1, 5)), "El árbol ya NO es idéntico con y sin escalado.")
chequear(all(BAL_[m] > 0 for m in MODELOS), "El balanceo ya no mejora el F1 medio en los 4 modelos.")
chequear(all(abs(BAL_[m]) > max(abs(ED_[m]), abs(OUT_[m])) for m in MODELOS), "El balanceo ya no es la decisión de mayor efecto en los 4 modelos.")
chequear(ED_["SVM"] > 0 and ED_["Redes Neuronales"] > 0, "El escalado ya no mejora SVM y redes.")
chequear(ED_["KNN"] < 0, "El escalado ya no empeora KNN en promedio.")
chequear(all(abs(OUT_[m]) < 0.02 for m in MODELOS), "El efecto medio de los outliers superó 0.02 de F1 en algún modelo.")
chequear(ranking[0][0] in ("Redes Neuronales", "SVM"), "El mejor F1 máximo ya no es de red o SVM.")
chequear(max(c8[["SVM", "Redes Neuronales"]]) > max(c8[["Árboles", "KNN"]]), "En el caso 8 SVM/red ya no superan a árbol y KNN.")
chequear(all(np.isclose(f1.loc[c, "Redes Neuronales"], f1.loc[1, "Redes Neuronales"]) for c in (1, 3)) and f1.loc[1, "Redes Neuronales"] < 0.1,
         "La red ya no colapsa en los casos 1 y 3.")
chequear(np.isclose(P["accuracy"].loc[1, "Redes Neuronales"], mod_share, atol=2e-3), "La exactitud de la red en el caso 1 ya no coincide con la proporción de Moderate.")
chequear(all(k == 1 for k in k_bal) and all(k > 1 for k in k_unb), "KNN ya no elige k=1 exactamente en los casos balanceados.")
chequear(roc["SVM"]["auc_macro"] > roc["Árboles"]["auc_macro"] and roc["Redes Neuronales"]["auc_macro"] > roc["Árboles"]["auc_macro"], "El AUC de SVM/red ya no supera al del árbol.")
chequear(roc["KNN"]["auc_macro"] == min(v["auc_macro"] for v in roc.values()), "KNN ya no tiene el AUC ROC más bajo.")
chequear(roc["KNN"]["auc_pr_micro"] > roc["Árboles"]["auc_pr_micro"], "El AUC-PR de KNN ya no supera al del árbol.")
chequear(all(v > 0.85 for v in adj.values()), "Ya no es cierto que casi todos los errores sean entre clases vecinas.")
chequear((F3["K óptimo (silhouette)"] == 2).all(), "K-Means ya no elige K=2 en los 8 casos.")
db_unb = F3[F3.Balanceo == "NO"]
db_bal = F3[F3.Balanceo == "SI"]
chequear((db_unb["DB clusters"] == 1).sum() == 3 and db_unb.loc[db_unb["#"] == 3, "DB clusters"].iloc[0] == 2, "DBSCAN ya no da 1 cluster en los casos desbalanceados 1, 5 y 7 (y 2 en el 3).")
chequear((db_bal["DB clusters"] >= 5).all(), "DBSCAN ya no encuentra ≥ 5 clusters en todos los casos balanceados.")
chequear(F3[F3.ED == "NO"]["KM silhouette"].mean() > F3[F3.ED == "SI"]["KM silhouette"].mean(), "El silhouette sin escalar ya no es mayor que con escalado.")
chequear(F3[F3.Balanceo == "SI"]["KM silhouette"].mean() > F3[F3.Balanceo == "NO"]["KM silhouette"].mean(), "El silhouette balanceado ya no supera al desbalanceado.")
for a, b in [(1, 3), (2, 4), (5, 7), (6, 8)]:
    chequear(F3.set_index("#").loc[b, "KM inertia"] > F3.set_index("#").loc[a, "KM inertia"], f"La inertia con outliers (caso {b}) ya no supera a la del caso {a}.")

ct_km = pd.read_csv(f"{D}/crosstab_caso8_kmeans.csv", index_col=0)
ct_db = pd.read_csv(f"{D}/crosstab_caso8_dbscan.csv", index_col=0)
# K-Means caso 8: el cluster "limpio" es el que concentra más Good + Satisfactory
km_clean = int(ct_km.loc[["Good", "Satisfactory"]].sum().idxmax())
km_dirty = 1 - km_clean
km_share = lambda cl, cls_: ct_km.loc[cls_, str(cl)].sum() / ct_km.loc[cls_].sum().sum()
chequear(km_share(km_dirty, ["Severe"]) > 0.7 and km_share(km_clean, ["Good"]) > 0.9, "El cruce de K-Means (caso 8) ya no separa limpio vs contaminado.")
db_noise_share = {c: ct_db.loc[c, "-1"] / ct_db.loc[c].sum() for c in CLASES}
db_cl = ct_db.drop(columns="-1")
db_big = int(db_cl.sum().idxmax())
sev_cl = int(db_cl.loc["Severe"].idxmax())
sev_purity = db_cl.loc["Severe", str(sev_cl)] / db_cl[str(sev_cl)].sum()
chequear(db_noise_share["Moderate"] > 0.5 and db_noise_share["Poor"] > 0.5 and db_noise_share["Good"] < 0.15, "El ruido de DBSCAN (caso 8) ya no se concentra en las clases intermedias.")
chequear((db_cl.loc[["Good", "Satisfactory"], str(db_big)].sum() / db_cl[str(db_big)].sum()) > 0.6, "El cluster grande de DBSCAN (caso 8) ya no es de aire limpio.")

# ----------------------------------------------------------------------------------------------------------------------
# Diagramas (se guardan en WORK)
# ----------------------------------------------------------------------------------------------------------------------
def diagrama():
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)
    cajas = [(0.1, f"city_day.csv\n({J['filas_originales']:,} filas)".replace(",", " ")), (1.85, "Subconjunto\nN según cédula\n+ semilla"),
             (3.6, "Partición estratificada\n70 / 15 / 15"), (5.35, "Imputación (train)\nmediana / moda"),
             (7.1, "Caso 1..8:\nrecorte / outliers 5 %\nbalanceo · One-Hot\nescalado (fit train)"), (9.0, "Modelos\nÁrbol · KNN · SVM · RN\nK-Means · DBSCAN")]
    for x, t in cajas:
        ax.add_patch(FancyBboxPatch((x, 1.2), 1.6, 1.4, boxstyle="round,pad=0.05", fc="#e8f0fe", ec="#3b5bdb"))
        ax.text(x + 0.8, 1.9, t, ha="center", va="center", fontsize=8)
    for x in [1.7, 3.45, 5.2, 6.95, 8.7]:
        ax.annotate("", xy=(x + 0.2, 1.9), xytext=(x - 0.02, 1.9), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(5.5, 0.55, "Validación y prueba reciben el mismo tratamiento en los 8 casos; imputador, codificador, escalador, SMOTE e inyección\nde outliers se ajustan únicamente con el conjunto de entrenamiento.",
            ha="center", fontsize=8.5, style="italic")
    plt.savefig(f"{WORK}/diagrama_pipeline.png", dpi=170, bbox_inches="tight"); plt.close()


def diagrama_red():
    capas = [(f"Entrada\n{n_feat} variables", "#e8f0fe"), ("Dense 64\nReLU · L2 1e-3", "#d3f9d8"), ("Dropout\n0.3", "#ffe3e3"),
             ("Dense 32\nReLU · L2 1e-3", "#d3f9d8"), ("Dropout\n0.3", "#ffe3e3"), ("Dense 6\nSoftmax", "#e5dbff")]
    fig, ax = plt.subplots(figsize=(11, 2.4)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 2.4)
    for i, (t, c) in enumerate(capas):
        x = 0.1 + i * 1.82
        ax.add_patch(FancyBboxPatch((x, 0.7), 1.4, 1.0, boxstyle="round,pad=0.04", fc=c, ec="#555"))
        ax.text(x + 0.7, 1.2, t, ha="center", va="center", fontsize=8.5)
        if i < len(capas) - 1:
            ax.annotate("", xy=(x + 1.80, 1.2), xytext=(x + 1.47, 1.2), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(5.5, 0.25, "Adam · categorical_crossentropy (one-hot) · lotes de 64 · hasta 200 épocas · EarlyStopping (val_loss, paciencia 10)", ha="center", fontsize=8.5, style="italic")
    plt.savefig(f"{WORK}/diagrama_red.png", dpi=170, bbox_inches="tight"); plt.close()


diagrama(); diagrama_red()


def img(name):
    """Ruta de una imagen: primero salidas/, luego los diagramas generados."""
    for base in (D, WORK):
        if os.path.exists(f"{base}/{name}"):
            return f"{base}/{name}"
    raise FileNotFoundError(name)


# ----------------------------------------------------------------------------------------------------------------------
# Tablas (devuelven especificaciones neutras)
# ----------------------------------------------------------------------------------------------------------------------
def tabla_fig1():
    hdr1 = [("#", 1, 2), ("Normalización", 1, 2), ("Outliers", 1, 2), ("Balanceo", 1, 2)] + [(m, 4, 1) for m in MODELOS]
    hdr2 = [("Acc", 1, 1), ("Prec", 1, 1), ("Rec", 1, 1), ("F1", 1, 1)] * 4
    rows = []
    for c in range(1, 9):
        cf = CASOS[c]
        rows.append([str(c), f"CC(SI) y ED({SI(cf['ED'])})", SI(cf["OUT"]), SI(cf["BAL"])] +
                    [(f"{P[k].loc[c, m]:.3f}" if k != "f1" else f"**{P[k].loc[c, m]:.3f}**") for m in MODELOS for k in ["accuracy", "precision", "recall", "f1"]])
    return dict(header=[hdr1, hdr2], rows=rows, widths=[0.6, 2.7, 1.6, 1.6] + [1.08] * 16, size=7.5, align_first=False)


def tabla_fig3():
    d = F3
    hdr = [[("#", 1, 2), ("Normalización", 1, 2), ("Outliers", 1, 2), ("Balanceo", 1, 2), ("K-Means", 2, 1), ("DBSCAN", 1, 1), ("Parámetros elegidos", 3, 1)],
           [("Inertia", 1, 1), ("Silhouette", 1, 1), ("Silhouette", 1, 1), ("K (codo)", 1, 1), ("Clusters / ruido", 1, 1), ("ε codo → ε usado", 1, 1)]]
    rows = []
    for _, r in d.iterrows():
        rows.append([str(int(r["#"])), f"CC(SI) y ED({r['ED']})", r["Outliers"], r["Balanceo"], f"{r['KM inertia']:,.0f}".replace(",", " "), f"{r['KM silhouette']:.3f}",
                     "n/a (1 cluster)" if pd.isna(r["DB silhouette"]) else f"{r['DB silhouette']:.3f}",
                     f"{int(r['K óptimo (silhouette)'])} ({int(r['K codo'])})", f"{int(r['DB clusters'])} / {r['DB ruido %']:.1f} %",
                     f"{r['DB eps codo']:.2f} → {r['DB eps usado']:.2f}"])
    return dict(header=hdr, rows=rows, widths=[0.7, 3.0, 1.8, 1.8, 2.6, 2.2, 2.8, 2.0, 2.8, 3.0], size=8.5, align_first=False)


def tabla_simple(df, widths=None, size=9, align_first=True):
    return dict(header=[[(str(c), 1, 1) for c in df.columns]], rows=[[str(x) for x in r] for r in df.values.tolist()], widths=widths, size=size, align_first=align_first)


def tabla_recall():
    filas = [[m] + [f"{rep(m).loc[c, 'recall']:.2f} / {rep(m).loc[c, 'f1-score']:.2f}" for c in CLASES] for m in MODELOS]
    return tabla_simple(pd.DataFrame(filas, columns=["Modelo"] + CLASES), widths=[3.2] + [2.2] * 6, size=8.5)


def tabla_efectos():
    filas = [[m] + [sg(ef[k][2][m]) for k in ["ED", "OUT", "BAL"]] for m in MODELOS]
    return tabla_simple(pd.DataFrame(filas, columns=["Modelo", "Escalado (SI − NO)", "Outliers (SI − NO)", "Balanceo (SI − NO)"]), widths=[3.8, 3.6, 3.6, 3.6])


def tabla_cruce(ct):
    ct = ct.copy()
    ct.columns = ["ruido" if str(c) == "-1" else f"cluster {c}" for c in ct.columns]
    t = ct.reset_index().rename(columns={"row_0": "Clase real", "index": "Clase real"})
    t.columns = ["Clase real"] + list(ct.columns)
    return tabla_simple(t, widths=[3.0] + [1.9] * len(ct.columns), size=8.5)


# ----------------------------------------------------------------------------------------------------------------------
# Contenido del informe (lista de bloques)
# Tipos: h1, h2, p, ul, ol, table(caption, spec), fig(file, caption, ancho_cm), figs2([(file, caption), ...]), eq, note, landscape(True/False)
# Marcado en línea: **negrita**  *cursiva*  `código`  ^superíndice^  ==resaltado==  <<url|texto>>  y referencias [[fig:clave]] / [[tab:clave]]
# ----------------------------------------------------------------------------------------------------------------------
B = []
add = B.append
h1 = lambda t: add(("h1", t))
h2 = lambda t: add(("h2", t))
p = lambda t: add(("p", t))
ul = lambda items: add(("ul", items))
ol = lambda items: add(("ol", items))
note = lambda t: add(("note", t))
eq = lambda t: add(("eq", t))
table = lambda key, cap, spec: add(("table", key, cap, spec))
fig = lambda key, f, cap, w=15.5: add(("fig", key, f, cap, w))
figs2 = lambda items: add(("figs2", items))

AUTORES = ", ".join(f"{n}^{k + 1}^" for k, n in enumerate(INTEGRANTES[:-1])) + f" y {INTEGRANTES[-1]}^{len(INTEGRANTES)}^"
dist_txt = ", ".join(f"{k}: {v}" for k, v in dist.items())
video_txt = f"<<{VIDEO_URL}|{VIDEO_URL}>>" if VIDEO_URL else "==[PENDIENTE: pegar aquí el enlace de YouTube del video pitch]=="
best = R.loc[R.f1.idxmax()]
tasa_out = J["outliers_train_natural_filas_pct"]
out_inertia = {(a, b): 100 * (F3.set_index("#").loc[b, "KM inertia"] / F3.set_index("#").loc[a, "KM inertia"] - 1) for a, b in [(1, 3), (2, 4), (5, 7), (6, 8)]}
sil_ed_no, sil_ed_si = F3[F3.ED == "NO"]["KM silhouette"].mean(), F3[F3.ED == "SI"]["KM silhouette"].mean()
sil_bal_no, sil_bal_si = F3[F3.Balanceo == "NO"]["KM silhouette"].mean(), F3[F3.Balanceo == "SI"]["KM silhouette"].mean()
d_out_case = {(a, b): f1.loc[b, "Árboles"] - f1.loc[a, "Árboles"] for a, b in [(1, 3), (2, 4)]}
gini_cases = [c for c in range(1, 9) if arbol[str(c)]["criterio"] == "gini"]
ent_cases = [c for c in range(1, 9) if arbol[str(c)]["criterio"] == "entropy"]
f3i = F3.set_index("#")

add(("title", dict(titulo="Trabajo Práctico 2", subtitulo="Calidad del aire en India: clasificación y agrupamiento del AQI", autores=AUTORES,
                   afil=f"Universidad Nacional de Colombia, Sede Medellín\nIntroducción a la Inteligencia Artificial (3010476) · Grupo {J['grupo']}, Equipo {J['equipo']}")))
add(("abstract",
     f"**Resumen.** Se trabajó con {J['n_registros']} registros de calidad del aire de 26 ciudades de India para clasificar el índice de calidad del aire (AQI) en seis categorías, "
     f"de *Good* a *Severe*, y luego agrupar los datos sin etiqueta. Se construyeron ocho versiones del conjunto de datos combinando escalado, un 5 % de valores atípicos y balanceo de clases; "
     f"en cada una se entrenaron un árbol de decisión, KNN, una SVM y una red neuronal y, con las mismas versiones y sin la etiqueta, K-Means y DBSCAN. "
     f"El F1 macro más alto fue {best.f1:.3f} ({best.modelo}, caso {int(best.caso)}). El balanceo de clases fue la decisión de mayor efecto en los cuatro modelos; "
     f"el escalado fue determinante para la red neuronal y la SVM, nulo para el árbol y desfavorable para KNN; y los valores atípicos inyectados modificaron poco el F1 medio. "
     f"En el aprendizaje no supervisado, K-Means separó el aire limpio del contaminado (K = 2 en los ocho casos) y DBSCAN solo encontró estructura de varios clusters con los datos balanceados."))
add(("p", f"**Enlace del video (pitch):** {video_txt}"))

# ------------------------------------------------------------------ 1
h1("Formulación del problema y dataset")
h2("Objetivo y alcance")
p("El trabajo implementa y compara técnicas de aprendizaje de máquinas sobre un dataset abierto: clasificación supervisada con árboles de decisión, KNN, máquinas de vectores de soporte (SVM) "
  "y redes neuronales; y agrupamiento no supervisado con K-Means y DBSCAN. Cada técnica se evalúa frente a ocho combinaciones de preprocesamiento (escalado, outliers y balanceo de clases), "
  "de modo que el análisis se centra en cómo cada decisión de preparación de datos cambia el comportamiento de cada modelo.")
h2("Fuente del dataset y justificación")
p(f"Se utilizó *Air Quality Data in India* (Kaggle, Rohan Rao; <<{DATASET_URL}|enlace al dataset>>), archivo `city_day.csv`: mediciones diarias de contaminantes en 26 ciudades de India "
  "entre 2015 y 2020. Se eligió porque (a) la variable objetivo `AQI_Bucket` tiene seis clases ordenadas por severidad; (b) combina variables numéricas y categóricas; "
  "(c) presenta valores nulos y valores extremos reales, por lo que el preprocesamiento exigido tiene un efecto verificable; y (d) los errores de clasificación tienen un costo distinto según su sentido: "
  "declarar *Moderate* cuando el aire está en *Severe* deja a la población sin alerta, mientras que el error contrario genera una alerta innecesaria.")
p("Las categorías corresponden a los rangos del índice de calidad del aire de India (CPCB) que definen `AQI_Bucket`:")
table("aqi", "Rangos del AQI que definen las seis clases de AQI_Bucket.",
      tabla_simple(pd.DataFrame([["Rango de AQI", "0–50", "51–100", "101–200", "201–300", "301–400", "401–500"]], columns=["AQI_Bucket"] + CLASES), size=9, align_first=True))
h2("Descripción de las variables")
table("vars", "Variables del conjunto de datos y uso en los modelos.", tabla_simple(pd.DataFrame([
    ["City", "Categórica nominal (26 ciudades)", "One-Hot"],
    ["Date", "Fecha (2015–2020)", "Se derivan `Season` (Winter, Summer, Monsoon, PostMonsoon; nominal, One-Hot) y `Year` (ordinal, por orden temporal)"],
    ["PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene", "Numéricas continuas (12)", "Imputación con mediana, recorte IQR y escalado"],
    ["AQI", "Numérica", "**Excluida**: `AQI_Bucket` se calcula directamente a partir de ella (fuga de información)"],
    ["AQI_Bucket", "Objetivo, 6 clases ordenadas", "Good < Satisfactory < Moderate < Poor < Very Poor < Severe"]],
    columns=["Variable", "Tipo", "Uso en el modelo"]), widths=[4.6, 4.2, 7.0], size=8.5))
h2("Verificación de requisitos del dataset")
ul([f"Cédula utilizada: **{J['cedula']}**. Último dígito distinto de cero = {ultimo_dig}, de modo que el tamaño del subconjunto es 5000 + {ultimo_dig} × 150 = **{J['n_registros']}** registros; "
    f"la semilla (*random state*) son los dos últimos dígitos distintos de cero: **{J['seed']}** (distinta de 42).",
    f"Registros del archivo original: {J['filas_originales']:,}; con etiqueta: {J['filas_etiquetadas']:,}. Subconjunto de trabajo: {J['n_registros']} (≥ 5000), muestreado al azar con `random_state={J['seed']}`.".replace(",", " "),
    f"Clases de la variable objetivo: 6 (≥ 4). Distribución del subconjunto: {dist_txt}.",
    "Variables numéricas (12) y categóricas (City, Season, Year).",
    f"Partición estratificada: entrenamiento {J['tamanos_split']['tr']}, validación {J['tamanos_split']['va']}, prueba {J['tamanos_split']['te']} (70/15/15)."])
eq(f"N = 5000 + d · 150 = 5000 + {ultimo_dig} · 150 = {J['n_registros']}")

# ------------------------------------------------------------------ 2
h1("Preprocesamiento")
fig("pipeline", "diagrama_pipeline.png", "Flujo general del preprocesamiento. Todo lo que aprende de los datos se ajusta solo con entrenamiento.", 16)
h2("Análisis exploratorio y valores nulos")
p(f"El subconjunto tiene {J['n_registros']} filas y 16 columnas (City, Season, Year, 12 contaminantes y la etiqueta). La columna con más nulos es Xylene ({nul['Xylene']:.1f} %), seguida de PM10 ({nul['PM10']:.1f} %), "
  f"NH3 ({nul['NH3']:.1f} %) y Toluene ({nul['Toluene']:.1f} %); el resto está por debajo de {max(v for k, v in nul.items() if k not in ('Xylene', 'PM10', 'NH3', 'Toluene')):.0f} %. "
  "Se imputó con la **mediana** (calculada solo con entrenamiento) porque los contaminantes tienen colas muy pesadas y la media se ve arrastrada por los picos; para las categóricas se usaría la moda. "
  f"En Xylene la imputación es discutible porque más de la mitad de la columna es un valor imputado; se mantuvo la variable para respetar el enunciado y se registra como limitación.")
corr_txt = ", ".join(f"{k} ({v})" for k, v in J["correlaciones_altas"].items())
p(f"Pares con correlación absoluta superior a 0.8: {corr_txt}. Es esperable: PM2.5 está contenido en PM10 y NOx incluye al NO. Para los árboles no es un problema; para KNN y SVM implica que esa información pesa más en la distancia.")
figs2([("eda_clases.png", "Distribución de clases del subconjunto."), ("eda_correlacion.png", "Matriz de correlación de las variables numéricas.")])
h2("Decisiones de preprocesamiento y justificación")
ul([f"**Conversión de categóricas (CC):** One-Hot para `City` y `Season`, porque no tienen un orden natural; codificación ordinal para `Year`, que sí tiene orden temporal. Es la decisión más discutible del dataset, "
    "ya que no trae variables ordinales evidentes y `Year` fue la opción menos forzada.",
    "**Escalado (ED):** `StandardScaler` ajustado solo con entrenamiento y aplicado a validación y prueba. Se prefirió a MinMax porque, con colas pesadas, un único valor extremo comprime al resto en un rango muy estrecho.",
    f"**Detección de outliers:** IQR con factor 1.5 sobre las 12 variables numéricas, con límites calculados en entrenamiento. Bajo ese criterio el **{tasa_out} %** de las filas de entrenamiento tiene al menos un valor atípico "
    f"(Xylene {J['outliers_por_variable_pct']['Xylene']} %, Toluene {J['outliers_por_variable_pct']['Toluene']} %), por lo que eliminarlos habría descartado más de la mitad de los datos.",
    "**Sin outliers:** los valores detectados se recortan (*winsorizing*) a los límites IQR.",
    f"**Con outliers:** al entrenamiento ya recortado se le inyecta sintéticamente el 5 % de las filas con un valor entre 1.5 y 4 IQR por encima del límite superior, en una variable numérica elegida al azar: "
    f"{casos_tr['3']['outliers']['n_inyectados']} filas en los casos 3 y 7 (de {n_train_unb}) y {casos_tr['4']['outliers']['n_inyectados']} filas en los casos 4 y 8 (de {n_train_bal}), es decir, "
    f"{casos_tr['3']['outliers']['pct']:.2f} % y {casos_tr['4']['outliers']['pct']:.2f} %.",
    f"**Desbalanceado:** submuestreo del entrenamiento hasta que la clase mayoritaria (Moderate) represente ≈ 70 % y las otras cinco queden iguales. Distribución resultante: "
    f"{', '.join(f'{k} {v}' for k, v in casos_tr['1']['dist_train'].items())} ({n_train_unb} filas; Moderate = {pct(casos_tr['1']['dist_train']['Moderate'] / n_train_unb, 1)}).",
    f"**Balanceado:** SMOTE sobre el entrenamiento natural hasta igualar las seis clases en {max(casos_tr['2']['dist_train'].values())} registros cada una ({n_train_bal} filas). Se eligió SMOTE porque interpola entre vecinos reales "
    "en lugar de repetir filas idénticas (lo que favorecería el sobreajuste); se aplica en un espacio estandarizado temporal para que las distancias no estén dominadas por las variables de mayor magnitud, y las variables One-Hot se redondean a 0/1 después."])
eq("límite inferior = Q1 − 1.5 · IQR        límite superior = Q3 + 1.5 · IQR        IQR = Q3 − Q1")
h2("Partición y reproducibilidad")
p(f"La partición estratificada 70/15/15 se realiza una sola vez con `random_state={J['seed']}`, de modo que el conjunto de prueba ({n_test} registros) es el mismo en los ocho casos, y la tabla de la Figura 1 es comparable fila contra fila. "
  "Imputación, codificación, escalado, inyección de outliers y balanceo se ajustan únicamente con el conjunto de entrenamiento. La misma semilla se usa en el muestreo, la partición, los modelos, SMOTE, la inyección de outliers y las inicializaciones de la red.")
note("**Tratamiento de validación y prueba.** Durante el desarrollo se comprobó que recortar los outliers solo en entrenamiento dejaba el conjunto de prueba con valores muy superiores a los vistos por los modelos, "
     "lo que penalizaba artificialmente a KNN, SVM y las redes. Por ello validación y prueba reciben siempre el mismo recorte, con los límites aprendidos en entrenamiento, en los ocho casos. "
     "Consecuencia: la prueba mide el desempeño con datos recortados y no ante outliers reales en predicción (véase la sección de limitaciones).")
note(f"**Efecto del tamaño del entrenamiento.** Los casos con balanceo no solo cambian la proporción de clases, también el tamaño del entrenamiento ({n_train_unb} frente a {n_train_bal} filas), "
     "de modo que parte de la mejora atribuida al balanceo puede deberse al mayor número de filas (sintéticas) disponibles.")

# ------------------------------------------------------------------ 3
h1("Aprendizaje supervisado")
h2("Modelos y configuración")
dep_txt = "; ".join(f"caso {c}: {arbol[str(c)]['criterio']}, profundidad {arbol[str(c)]['max_depth']}" for c in range(1, 9))
table("cfg", "Configuración de las cuatro técnicas supervisadas.", tabla_simple(pd.DataFrame([
    ["Árbol de decisión", f"Se evaluaron los criterios Gini y Entropía con `max_depth` ∈ {{2, 3, 4, 5, 6, 8, 10, 12, 15, 20}} y se eligió la combinación de mayor F1 macro en validación. Resultado por caso: {dep_txt}."],
    ["KNN", f"k impar entre 1 y 25, elegido por accuracy medio de validación cruzada de 5 folds. k elegido por caso: {', '.join(str(knn_k[str(c)]) for c in range(1, 9))}."],
    ["SVM", "Kernel RBF, C = 1, gamma = 'scale'. Para las curvas ROC/PR se usan los puntajes de `decision_function`, que no son probabilidades calibradas."],
    ["Red neuronal (MLP)", f"Dense(64, ReLU, L2 1e-3) → Dropout 0.3 → Dense(32, ReLU, L2 1e-3) → Dropout 0.3 → Dense(6, softmax). Adam, `categorical_crossentropy` con etiquetas one-hot, lotes de 64, hasta 200 épocas, "
     f"validación en `validation_data`. Regularización y callback vistos en clase: L2, Dropout y `EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)`. El entrenamiento se detuvo entre {min(nn_ep.values())} y {max(nn_ep.values())} épocas según el caso."]],
    columns=["Técnica", "Configuración"]), widths=[3.4, 12.4], size=8.5))
fig("red", "diagrama_red.png", "Arquitectura de la red neuronal (MLP) usada en los ocho casos.", 16)
p("La configuración de la red se obtuvo de forma incremental: se partió de una arquitectura pequeña y se aumentó la regularización hasta que las curvas de entrenamiento y validación dejaron de separarse (véase la curva del caso 8). "
  "Las métricas son accuracy y precision, recall y F1 **macro** (promedio simple por clase), de modo que clases pequeñas como *Severe* pesan lo mismo que *Moderate*.")
eq("F1 macro = (1/6) · Σ  2 · precisión_c · recall_c / (precisión_c + recall_c)")
add(("landscape", True))
h2("Resultados: Figura 1 del enunciado")
table("fig1", f"Resultados de los modelos supervisados en los ocho casos (Figura 1 del enunciado). ED = escalado; CC siempre aplicada. Métricas sobre el mismo conjunto de prueba ({n_test} registros); en negrita, el F1 macro.", tabla_fig1())
add(("landscape", False))
h2("Gráficas solicitadas")
figs2([("f1_max_algoritmo.png", "F1 macro máximo por algoritmo."), ("f1_medio_ed.png", "F1 macro medio por escalado y algoritmo.")])
figs2([("f1_medio_outliers.png", "F1 macro medio por manejo de outliers y algoritmo."), ("f1_medio_balanceo.png", "F1 macro medio por balanceo y algoritmo.")])
p("La [[tab:efectos]] resume el efecto medio de cada decisión sobre el F1 macro (promedio de los cuatro casos con el factor en SI menos el promedio de los cuatro casos con el factor en NO).")
table("efectos", "Efecto medio de cada decisión de preprocesamiento sobre el F1 macro.", tabla_efectos())
ul([f"**Balanceo:** es la decisión de mayor efecto en los cuatro modelos ({', '.join(f'{m} {sg(BAL_[m])}' for m in MODELOS)}).",
    f"**Escalado:** decisivo para la red ({sg(ED_['Redes Neuronales'])}) y la SVM ({sg(ED_['SVM'])}), nulo para el árbol ({sg(ED_['Árboles'])}) y desfavorable en promedio para KNN ({sg(ED_['KNN'])}).",
    f"**Outliers:** efecto medio pequeño en todos los modelos (de {sg(min(OUT_))} a {sg(max(OUT_))}).",
    f"**Ranking por F1 máximo:** {' > '.join(f'{m} ({v[0]:.3f}, caso {v[1]})' for m, v in ranking)}."])
h2("Análisis de resultados")
p(f"**Mejor modelo.** El F1 máximo general lo obtiene {ranking[0][0]} ({fmt(ranking[0][1][0])}, caso {ranking[0][1][1]}), seguido de {ranking[1][0]} ({fmt(ranking[1][1][0])}, caso {ranking[1][1][1]}), "
  f"{ranking[2][0]} ({fmt(ranking[2][1][0])}, caso {ranking[2][1][1]}) y {ranking[3][0]} ({fmt(ranking[3][1][0])}, caso {ranking[3][1][1]}). En el caso 8, con todo el preprocesamiento aplicado, "
  f"SVM ({fmt(c8['SVM'])}) y red ({fmt(c8['Redes Neuronales'])}) superan a árbol ({fmt(c8['Árboles'])}) y KNN ({fmt(c8['KNN'])}), que quedan prácticamente empatados. Las diferencias entre los mejores modelos son del orden de 0.01 "
  "y provienen de una sola corrida por caso, por lo que no deben leerse como una superioridad concluyente.")
p(f"**Sensibilidad a la escala.** El árbol produjo *exactamente* los mismos resultados con y sin escalado (casos 1 = 5, 2 = 6, 3 = 7 y 4 = 8): divide por umbrales sobre una variable a la vez y el escalado, al ser una transformación monótona, solo desplaza el umbral. "
  f"La red neuronal y la SVM dependen de la escala ({sg(ED_['Redes Neuronales'])} y {sg(ED_['SVM'])} de F1 medio): con unidades crudas PM10 (cientos) y CO (unidades) no pesan por igual en el kernel RBF ni en los pesos de la red. "
  f"El caso más marcado es la red neuronal sin escalar y con entrenamiento desbalanceado (casos 1 y 3): su F1 es {fmt(f1.loc[1, 'Redes Neuronales'], 4)} y su accuracy ({P['accuracy'].loc[1, 'Redes Neuronales']:.4f}) coincide con la proporción de *Moderate* en el conjunto de prueba "
  f"({pct(mod_share, 2)}), lo que indica que la red colapsó a predecir siempre la clase mayoritaria. "
  f"KNN fue la excepción al patrón habitual ({sg(ED_['KNN'])} de F1 medio): escalar lo favorece con el entrenamiento balanceado ({sg(f1.loc[6, 'KNN'] - f1.loc[2, 'KNN'])} y {sg(f1.loc[8, 'KNN'] - f1.loc[4, 'KNN'])} en los casos 2→6 y 4→8) "
  f"y lo perjudica con el desbalanceado ({sg(f1.loc[5, 'KNN'] - f1.loc[1, 'KNN'])} y {sg(f1.loc[7, 'KNN'] - f1.loc[3, 'KNN'])} en 1→5 y 3→7). No se realizó un experimento que explique esa asimetría, por lo que se deja como observación sin causa comprobada.")
p(f"**Outliers.** El efecto medio fue pequeño en los cuatro modelos (entre {sg(min(OUT_))} y {sg(max(OUT_))}), aunque en comparaciones puntuales el árbol varió hasta {max(abs(v) for v in d_out_case.values()):.3f} de F1 "
  f"(caso 1→3: {sg(d_out_case[(1, 3)])}; caso 2→4: {sg(d_out_case[(2, 4)])}), con signos distintos según el balanceo, lo que sugiere variabilidad más que un efecto sistemático. "
  "Dos factores explican la sensibilidad reducida: se inyectó un 5 % de valores moderados (1.5 a 4 IQR por encima del límite) y el conjunto de prueba va recortado, de modo que se mide cuánto perjudica *entrenar* con ruido, no cuánto perjudica recibirlo al predecir.")
p(f"**Desbalance y por qué el accuracy engaña.** El balanceo fue la decisión de mayor efecto ({sg(BAL_['Árboles'])} en árbol, {sg(BAL_['KNN'])} en KNN, {sg(BAL_['SVM'])} en SVM y {sg(BAL_['Redes Neuronales'])} en red). "
  "Con un entrenamiento dominado por *Moderate* (≈ 70 %), los modelos aprenden que predecir esa clase casi siempre es rentable y atienden mal a las clases pequeñas; un clasificador que respondiera siempre *Moderate* "
  "tendría 70 % de accuracy sobre ese entrenamiento sin haber aprendido nada, y el caso extremo se observó en la red de los casos 1 y 3. En este trabajo el conjunto de prueba conserva la distribución natural "
  f"(*Moderate* ≈ {pct(mod_share)}), por lo que el accuracy global se infla menos de lo que ocurriría con una prueba 70/30; el efecto se ve con mayor claridad en el recall por clase y en el F1 macro, que son las métricas reportadas.")
p("**Costo de los errores y métrica priorizada.** En calidad del aire un falso negativo (declarar *Moderate* cuando el aire está en *Severe*) deja a la población sin alerta, mientras que un falso positivo (alertar de más) genera molestias y costos pero no daño directo a la salud. "
  "Por ello se prioriza el **recall de las clases altas** (Poor, Very Poor y Severe) y se usa el **F1 macro** como resumen, ya que evita que *Moderate* y *Satisfactory* oculten los fallos en las clases poco frecuentes.")
h2("Matrices de confusión y reporte por clase (caso 8)")
fig("conf", "matrices_confusion_caso8.png", "Matrices de confusión de las cuatro técnicas en el caso 8 (conjunto de prueba).", 15)
table("recall", "Recall / F1 por clase en el caso 8, conjunto de prueba.", tabla_recall())
p(f"El recall de *Severe* en el caso 8 fue {', '.join(f'{m} {sev[m]:.2f}' for m in MODELOS)}; la clase más difícil para los cuatro modelos es *Poor* (F1 entre {min(poor_f1.values()):.2f} y {max(poor_f1.values()):.2f}). "
  f"Entre el {min(adj.values()) * 100:.0f} % y el {max(adj.values()) * 100:.0f} % de los errores de cada modelo ocurre entre clases vecinas de la escala (por ejemplo, *Very Poor* predicho como *Severe*), "
  "coherente con que el AQI es un continuo dividido por umbrales. Los reportes completos por clase (precisión, recall y F1) están en `salidas/reporte_clases_caso8_*.csv`.")
h2("Curvas ROC y Precision-Recall (caso 8)")
fig("roc", "roc_pr_caso8.png", "Curvas ROC y Precision-Recall (promedio micro, uno contra todos) de las cuatro técnicas en el caso 8.", 16)
p(f"El AUC macro (uno contra todos) fue: red {roc['Redes Neuronales']['auc_macro']:.3f}, SVM {roc['SVM']['auc_macro']:.3f}, árbol {roc['Árboles']['auc_macro']:.3f} y KNN {roc['KNN']['auc_macro']:.3f}. "
  f"El AUC-PR micro fue: red {roc['Redes Neuronales']['auc_pr_micro']:.3f}, SVM {roc['SVM']['auc_pr_micro']:.3f}, KNN {roc['KNN']['auc_pr_micro']:.3f} y árbol {roc['Árboles']['auc_pr_micro']:.3f}. "
  "Red y SVM ordenan claramente mejor los casos que árbol y KNN. Un árbol de profundidad limitada reparte pocas probabilidades distintas (una por hoja), por lo que su curva es escalonada y tiene poca resolución para ordenar; "
  f"con k = 1, las «probabilidades» de KNN son 0 o 1, lo que explica su AUC ROC, el más bajo. El AUC-PR de KNN supera al del árbol, pero con tan pocos umbrales la curva tiene muy pocos puntos y el área por trapecios resulta poco informativa, "
  "así que para KNN se confía más en el AUC ROC y en el F1. Los puntajes de la SVM no son probabilidades calibradas, de modo que las curvas se comparan como rankings.")
h2("Visualización por técnica")
p(f"**Árbol de decisión.** Las variables más importantes en el caso 8 son: {', '.join(f'{k} ({v:.2f})' for k, v in arbol['8']['top_features'].items())}. "
  "La figura incluye la importancia de características, la curva de `max_depth` frente al accuracy de entrenamiento y validación para Gini y Entropía, y el árbol obtenido con Gini (primeros niveles); "
  "las reglas completas están en `salidas/reglas_arbol_casoN.txt`.")
fig("arbol", "arbol_caso8.png", "Árbol de decisión (caso 8): importancia de características, `max_depth` vs accuracy (Gini y Entropía) y primeros niveles del árbol con Gini. Los ocho casos están en el Anexo B.", 16.3)
p("**Lectura en el dominio.** Que PM2.5 y PM10 sean las variables dominantes es coherente con el dominio: el AQI se calcula a partir de subíndices por contaminante y el material particulado suele ser el de mayor peso. "
  "La regla raíz del árbol Gini del caso 8 divide por PM10 (valor estandarizado ≤ −0.38); con PM10 muy bajo y CO bajo el árbol se orienta a *Good*, y con PM2.5 alto y CO alto a *Very Poor*. "
  "Un hallazgo menos cómodo es que `City_Ahmedabad` aparece como la tercera variable más importante: el árbol usa esa ciudad como atajo porque presenta niveles persistentemente altos. Es útil para predecir, "
  "pero es información de ubicación y no de contaminación, y no implica que estar en esa ciudad cause un peor AQI.")
p(f"**Subajuste y sobreajuste.** La exactitud de entrenamiento crece hasta cerca de 1.0 al aumentar `max_depth`, mientras la de validación se estabiliza alrededor de 0.70 a partir de la profundidad 6; la diferencia entre ambas curvas es el sobreajuste. "
  "Con profundidades de 2 a 3 ambas son bajas (subajuste). Gini y Entropía dieron curvas muy parecidas; "
  f"la selección por F1 de validación favoreció Entropía en los casos {', '.join(map(str, ent_cases))} y Gini en los casos {', '.join(map(str, gini_cases))}. "
  "El árbol que se dibuja es el mejor con criterio Gini, como pide el enunciado, aunque no coincida con el modelo evaluado en los casos donde ganó Entropía.")
p(f"**KNN.** En los casos balanceados la validación cruzada eligió k = {', '.join(map(str, k_bal))} (siempre 1), con un accuracy de validación cruzada de ≈ 0.86 en el caso 8, mientras que en prueba el accuracy fue "
  f"{P['accuracy'].loc[8, 'KNN']:.2f}. La diferencia es un efecto de SMOTE: los puntos sintéticos son interpolaciones de puntos reales y, al repartirse entre folds, un punto de validación tiene casi un «gemelo» en entrenamiento, "
  "de modo que k = 1 parece excelente en validación cruzada sin serlo. Es una limitación reconocida: para los casos balanceados el k elegido es optimista.")
fig("knn", "knn_caso8.png", "KNN (caso 8): k vs accuracy (validación cruzada de 5 folds) y vecinos de tres puntos de prueba en PCA 2D. Los vecinos se calculan en el espacio completo; PCA solo se usa para dibujar.", 16)
p("**SVM.** Las regiones de decisión se organizan en franjas que van del aire más limpio al más contaminado y se solapan en la zona central, donde conviven *Moderate* y *Poor*. "
  "Con dos componentes se pierde información, por lo que el solapamiento del dibujo exagera el del modelo real.")
fig("svm", "svm_regiones_caso8.png", "Regiones de decisión de la SVM en PCA 2D (caso 8). Se reentrena una SVM sobre la proyección solo para poder dibujar; el modelo evaluado usa todas las variables.", 11)
p(f"**Red neuronal.** En el caso 8 la pérdida de validación se aplana hacia la época 15 y el entrenamiento se detuvo en la época {nn_ep['8']} restaurando los mejores pesos; "
  "la brecha entre entrenamiento y validación en accuracy es moderada gracias al Dropout y a la regularización L2.")
fig("nn", "nn_curvas_caso8.png", "Curvas de pérdida y accuracy de la red neuronal (caso 8).", 15)
h2("Tres datos de prueba (caso 8)")
t8 = T3[T3.caso == 8].drop(columns="caso"); t8.columns = ["Dato de prueba", "Clase real"] + MODELOS
table("tres_sup", "Tres datos de prueba, caso 8: clase real y predicción de cada técnica.", tabla_simple(t8, widths=[2.6, 3.0, 2.5, 2.5, 2.5, 2.7], size=8.5))
p("Los mismos tres registros se evaluaron en los ocho casos (Anexo A). Tres ejemplos ilustran el uso de los modelos pero no permiten concluir sobre su calidad; la evidencia es la tabla de la Figura 1.")
h2("Conclusiones del módulo supervisado")
ul([f"Con el preprocesamiento completo (caso 8), SVM y red alcanzan F1 macro de {fmt(c8['SVM'])} y {fmt(c8['Redes Neuronales'])}, y árbol y KNN de {fmt(c8['Árboles'])} y {fmt(c8['KNN'])}. La diferencia más grande se debe a la preparación de los datos y no a la elección del algoritmo.",
    "El árbol es el más robusto al preprocesamiento (no depende del escalado y reacciona poco a los outliers) y el más fácil de explicar; SVM y red necesitan datos escalados y ordenan mejor los casos según ROC/PR.",
    "Para este dominio, SVM o red neuronal sobre datos escalados y balanceados son preferibles si importa el ranking de riesgo, y el árbol si importa explicar las reglas. En ambos casos conviene vigilar el recall de *Severe* y no el accuracy."])

# ------------------------------------------------------------------ 4
h1("Aprendizaje no supervisado")
h2("Construcción de los casos")
p("Se utiliza el mismo dataset sin la variable objetivo. Las ocho versiones de la Figura 2 son exactamente los conjuntos de entrenamiento del módulo supervisado (mismos registros, mismo recorte o inyección de outliers, "
  "mismo balanceo, One-Hot y escalado) con la etiqueta descartada. Así, «balanceado» significa que los datos incluyen los puntos sintéticos de SMOTE y «desbalanceado» corresponde al submuestreo con 70 % de *Moderate*. "
  "La etiqueta original solo se usa para la validación externa (cruce de cada cluster con la clase real) y para interpretar los clusters, nunca para entrenar.")
h2("Configuración")
ul([f"**K-Means:** inicialización k-means++, `n_init=10`, `random_state={J['seed']}`, K de 2 a 10. Se reportan inertia (codo) y silhouette (calculado sobre una muestra de 3000 puntos); el K elegido es el de mayor silhouette y el K del codo se muestra al lado.",
    f"**DBSCAN:** MinPts = 2 × número de características = {2 * n_feat} (con {n_feat} columnas tras el One-Hot). ε se toma del codo de la curva de k-distancia (k = MinPts). El silhouette se calcula sin los puntos de ruido y requiere al menos dos clusters."])
note("**Ajuste de ε.** Con el ε del codo, DBSCAN dejaba todos los datos en un solo cluster en varios casos (silhouette no calculable). Cuando ocurrió, se exploraron 12 valores de ε entre 0.3 y 0.95 veces el del codo y se eligió el de mayor silhouette "
     "con al menos dos clusters y a lo sumo 50 % de ruido; si ninguno cumplía, se conservó el ε del codo. La [[tab:fig3]] muestra ambos valores. Es una decisión propia del equipo, no del procedimiento visto en clase.")
add(("landscape", True))
h2("Resultados: Figura 3 del enunciado")
table("fig3", "Resultados de K-Means y DBSCAN en los ocho casos (Figura 3 del enunciado) y parámetros elegidos.", tabla_fig3())
add(("landscape", False))
fig("sil", "silhouette_por_caso.png", "Silhouette por caso para K-Means (izquierda) y DBSCAN (derecha; 0 indica que no es calculable).", 15)
fig("param", "kmeans_dbscan_seleccion_parametros.png", "Selección de parámetros por caso: codo (inertia), silhouette vs K y curva de k-distancia (línea roja: ε del codo; verde punteada: ε usado).", 13)
h2("Análisis de resultados")
p(f"**K-Means.** El silhouette eligió K = 2 en los ocho casos, mientras que el codo sugirió entre {int(F3['K codo'].min())} y {int(F3['K codo'].max())}. La diferencia es informativa: las seis clases del AQI son más un continuo "
  "(cada clase es un tramo de la misma escala) que seis grupos separados, así que el único corte limpio es «aire más limpio» frente a «aire más contaminado». "
  f"Al cruzar con la etiqueta en el caso 8 (sobre el entrenamiento balanceado, con {max(casos_tr['2']['dist_train'].values())} registros por clase), el cluster {km_clean} reúne el {pct(km_share(km_clean, ['Good']))} de *Good*, el {pct(km_share(km_clean, ['Satisfactory']))} de *Satisfactory* y el {pct(km_share(km_clean, ['Moderate']))} de *Moderate*, "
  f"mientras que el cluster {km_dirty} reúne el {pct(km_share(km_dirty, ['Severe']))} de *Severe* y el {pct(km_share(km_dirty, ['Very Poor']))} de *Very Poor*; *Poor* queda repartida ({pct(km_share(km_dirty, ['Poor']))} en el cluster {km_dirty}). "
  "La partición tiene sentido para el dominio, aunque con K = 2 no pueden recuperarse las seis clases.")
table("ct_km", "Cruce de clusters de K-Means con la clase real, caso 8 (solo interpretación; la etiqueta no intervino en el entrenamiento).", tabla_cruce(ct_km))
p(f"**Efecto del escalado.** La inertia no es comparable entre casos con y sin escalado: sin escalar es del orden de millones ({f3i.loc[1, 'KM inertia']:,.0f} en el caso 1) y escalada de decenas de miles ({f3i.loc[5, 'KM inertia']:,.0f} en el caso 5), simplemente por las unidades. ".replace(",", " ") +
  f"El silhouette sin escalar fue en promedio mayor ({sil_ed_no:.3f} frente a {sil_ed_si:.3f}), pero no debe leerse como una mejora: sin escalar, las variables de mayor magnitud dominan la distancia euclidiana y el agrupamiento depende casi de ellas, "
  "lo que facilita separar dos grupos y eleva el silhouette. Este último razonamiento es una explicación plausible que no se comprobó con un experimento adicional.")
p(f"**Efecto de los outliers.** La inyección del 5 % aumentó la inertia en {out_inertia[(1, 3)]:.0f} % sin escalar y desbalanceado (caso 1→3), {out_inertia[(2, 4)]:.0f} % sin escalar y balanceado (2→4), {out_inertia[(5, 7)]:.0f} % con escalado y desbalanceado (5→7) y "
  f"{out_inertia[(6, 8)]:.0f} % con escalado y balanceado (6→8), mientras que el silhouette cambió poco. Es lo esperado: la inertia suma distancias al cuadrado y unos pocos puntos lejanos la inflan, pero el centroide se desplaza poco con solo 5 % de datos.")
p(f"**Efecto del balanceo.** El silhouette medio de K-Means subió de {sil_bal_no:.3f} a {sil_bal_si:.3f} con los datos balanceados. No debe interpretarse como una mejora real de la estructura: SMOTE rellena el espacio entre puntos reales con puntos interpolados, lo que vuelve los grupos más compactos e infla el silhouette.")
nz = {c: f3i.loc[c] for c in range(1, 9)}
p(f"**DBSCAN.** Con el entrenamiento desbalanceado, DBSCAN dejó un único cluster en los casos 1, 5 y 7 (ruido de {f3i.loc[1, 'DB ruido %']:.1f} %, {f3i.loc[5, 'DB ruido %']:.1f} % y {f3i.loc[7, 'DB ruido %']:.1f} %) y, en el caso 3, "
  f"2 clusters tras reducir ε de {f3i.loc[3, 'DB eps codo']:.1f} a {f3i.loc[3, 'DB eps usado']:.1f}, a costa de {f3i.loc[3, 'DB ruido %']:.0f} % de ruido. Con {n_train_unb} puntos en {n_feat} dimensiones y MinPts = {2 * n_feat}, la densidad es baja y se necesita un ε grande para que haya núcleos; a ese ε todo se conecta. "
  f"Con los datos balanceados, que tienen más de cuatro veces más puntos, aparecieron entre {int(db_bal['DB clusters'].min())} y {int(db_bal['DB clusters'].max())} clusters, con {db_bal['DB ruido %'].min():.0f} a {db_bal['DB ruido %'].max():.0f} % de ruido. "
  f"El caso 6 conserva el ε del codo y deja {f3i.loc[6, 'DB ruido %']:.1f} % de ruido, por lo que su silhouette ({f3i.loc[6, 'DB silhouette']:.3f}) se calcula sobre una fracción pequeña de los puntos. "
  "El silhouette de DBSCAN excluye el ruido y se obtiene con ε distintos en cada caso, por lo que no es comparable con el de K-Means ni entre casos.")
ct_cap = f"Cruce de clusters de DBSCAN con la clase real, caso 8 (−1 = ruido)."
table("ct_db", ct_cap, tabla_cruce(ct_db))
p(f"En el caso 8 el cruce muestra qué encontró DBSCAN: un cluster grande de aire limpio (el cluster {db_big} reúne el {pct(ct_db.loc['Good', str(db_big)] / ct_db.loc['Good'].sum())} de *Good* y el {pct(ct_db.loc['Satisfactory', str(db_big)] / ct_db.loc['Satisfactory'].sum())} de *Satisfactory*), "
  f"un cluster dominado por *Severe* (el cluster {sev_cl}, con {pct(sev_purity)} de *Severe*) y varios más pequeños de *Poor*, *Very Poor* y *Severe*. El ruido concentra las clases intermedias: "
  f"{pct(db_noise_share['Moderate'])} de *Moderate* y {pct(db_noise_share['Poor'])} de *Poor* frente a {pct(db_noise_share['Good'])} de *Good*. DBSCAN identifica los extremos densos, pero la franja intermedia es un continuo sin vacíos de densidad que delimiten clusters.")
h2("Comparación entre K-Means y DBSCAN")
table("cmp", "Comparación entre K-Means y DBSCAN.", tabla_simple(pd.DataFrame([
    ["Forma de los clusters", "Esféricos o convexos (distancia al centroide)", "Arbitraria; en el PCA se ven regiones alargadas y manchas densas"],
    ["Definir K", "Debe fijarse (codo y silhouette); aquí resultó 2", "No se define K, pero hay que acertar con ε y MinPts, más delicados que K"],
    ["Ruido y outliers", "Todo punto cae en algún cluster; los outliers inflan la inertia", f"Marca el ruido explícitamente; con datos balanceados fue de {db_bal['DB ruido %'].min():.0f} a {db_bal['DB ruido %'].max():.0f} %"],
    ["Densidades variables", "Indiferente a la densidad, solo mira distancia", "Sufre: un único ε no sirve para zonas densas y dispersas a la vez (aquí, un cluster o mucho ruido)"],
    [f"Alta dimensión ({n_feat} columnas)", "Aguanta mejor", "La distancia pierde contraste y la densidad se diluye"]], columns=["Aspecto", "K-Means", "DBSCAN"]), widths=[3.4, 6.0, 6.4], size=8.5))
h2("Visualización de clusters en 2D (PCA)")
fig("clusters", "clusters_caso8.png", "Caso 8: K-Means (K = 2, centroides en rojo) y DBSCAN (ruido en gris) proyectados con PCA. Los ocho casos están en el Anexo B.", 16)
p("K-Means corta la nube en dos mitades a lo largo del primer componente, que esencialmente ordena de aire más limpio a más contaminado; DBSCAN recoge las zonas densas y deja en gris la franja intermedia. "
  "Los clusters son interpretables en términos del dominio (limpio frente a contaminado), con la salvedad de que PCA con dos componentes solo da una visión parcial de un espacio de " + f"{n_feat} dimensiones.")
h2("Tres datos de prueba (caso 8)")
u8 = T3u[T3u.caso == 8].drop(columns="caso"); u8.columns = ["Dato de prueba", "Clase real (solo referencia)", "Cluster K-Means", "Cluster DBSCAN (−1 = ruido)"]
table("tres_uns", "Tres datos de prueba, caso 8: cluster asignado por K-Means y DBSCAN.", tabla_simple(u8, widths=[3.0, 4.4, 3.4, 4.4], size=8.5))
p("K-Means asigna cada dato al centroide más cercano; DBSCAN no tiene `predict`, de modo que cada dato toma el cluster del punto núcleo más cercano si queda a una distancia ≤ ε y se marca como ruido en caso contrario.")
h2("Conclusiones del módulo no supervisado")
ul(["Los datos no presentan seis grupos naturales: el AQI es una escala continua dividida por umbrales. El agrupamiento recupera bien el eje «limpio ↔ contaminado» y mal las clases intermedias.",
    "K-Means es estable y fácil de interpretar pero obliga a particionar todos los datos; DBSCAN es más explícito con el ruido pero muy sensible a ε, MinPts y al tamaño del conjunto.",
    "Las comparaciones entre casos con y sin escalado, o con y sin balanceo, requieren cautela: la inertia depende de las unidades y SMOTE infla la cohesión."])

# ------------------------------------------------------------------ 5
h1("Conclusiones generales y limitaciones")
ul(["Lo que más movió los resultados fue la preparación de los datos (balanceo y escalado) y que el conjunto de prueba recibiera el mismo tratamiento que el de entrenamiento; ningún algoritmo ganó por sí solo.",
    "El patrón se repite en ambos módulos: árbol y K-Means son robustos y explicables; SVM, red neuronal y DBSCAN rinden o fallan según el preprocesamiento.",
    "**Limitaciones.** (1) Una sola corrida por caso, sin repeticiones con varias semillas: diferencias de ≈ 0.01 no son concluyentes. (2) El tamaño del entrenamiento cambia entre balanceado y desbalanceado. "
    f"(3) Xylene tiene {nul['Xylene']:.0f} % de nulos imputados con la mediana. (4) El k de KNN en los casos balanceados es optimista por el efecto de SMOTE en la validación cruzada. "
    "(5) `Year` como variable ordinal es una decisión discutible. (6) El conjunto de prueba se recorta con límites de entrenamiento, por lo que no mide el desempeño ante outliers reales. "
    "(7) El ajuste de ε en DBSCAN es una decisión propia, descrita en la sección de configuración del módulo no supervisado y, en el caso 6, deja una proporción alta de ruido. "
    "(8) El cruce de clusters con la etiqueta del caso 8 se calcula sobre un entrenamiento que incluye puntos sintéticos."])

# ------------------------------------------------------------------ 6
h1("Origen de las técnicas respecto al material del curso")
p("Para la sustentación se identifica qué herramientas provienen del material de clase y cuáles se incorporaron por exigencia del enunciado o por decisión del equipo.")
table("curso", "Origen de cada técnica o decisión.", tabla_simple(pd.DataFrame([
    ["Partición de datos", "Entrenamiento 70 % / validación 15 % / prueba 15 %; la validación decide hiperparámetros y la prueba se usa al final", "Clase de redes neuronales"],
    ["Escalado", "StandardScaler ajustado en entrenamiento", "Taller 8; clase de K-Means"],
    ["Árboles", "`DecisionTreeClassifier`, Gini frente a Entropía, `max_depth`, `feature_importances_`, `plot_tree`", "Clase de árboles de decisión"],
    ["Métricas", "Accuracy, precisión, recall, F1 macro, matriz de confusión, ROC/AUC y Precision-Recall/AUC-PR", "Clase de métricas; taller 7"],
    ["Red neuronal", "Dense, ReLU, softmax, entropía cruzada categórica con etiquetas one-hot, Adam, L2, Dropout, EarlyStopping", "Clase de redes neuronales; taller 8"],
    ["K-Means", "Inicialización k-means++, codo con SSE (inertia), silhouette", "Clase de K-Means; taller 7"],
    ["DBSCAN", "ε con la curva de k-distancia, MinPts = 2 × nº de características, silhouette sin ruido", "Clase de DBSCAN; taller 7"],
    ["KNN y SVM", "KNN con k por validación cruzada de 5 folds; SVM con kernel RBF", "Exigidos por el enunciado"],
    ["SMOTE, submuestreo, recorte IQR, PCA para graficar", "Balanceo, tratamiento de outliers y visualización", "Los nombra el enunciado"],
    ["Búsqueda de un ε menor en DBSCAN", "Cuando el ε del codo deja un solo cluster", "Decisión del equipo"]], columns=["Tema", "Qué se usó", "Origen"]), widths=[3.6, 8.2, 4.0], size=8.5))

# ------------------------------------------------------------------ 7
h1("Reproducibilidad y ejecución")
ol(["Abrir `practica2.ipynb` en Google Colab y subir `city_day.csv` (o dejar que `kagglehub` lo descargue).",
    "Editar únicamente la celda de configuración (`CEDULA`, `GRUPO`, `EQUIPO`); el tamaño del subconjunto y la semilla se derivan automáticamente.",
    "Ejecutar todas las celdas (*Entorno → Ejecutar todo*). Al final se genera `salidas.zip` con las tablas y figuras de este documento.",
    "Descomprimir `salidas.zip` junto a `build_informe.py` y ejecutar `python build_informe.py` para regenerar este informe en formato Word; si alguna afirmación del texto deja de cumplirse con los nuevos números, la consola imprime «REVISAR TEXTO»."])

# ------------------------------------------------------------------ 8
h1("Enlaces")
ul([f"Video de sustentación (máx. 5 min; participan todos los integrantes): {video_txt}", f"Conjunto de datos original: <<{DATASET_URL}|{DATASET_URL}>>"])

# ------------------------------------------------------------------ Anexos
add(("pagebreak",))
add(("h1u", "Anexo A. Tres datos de prueba en los ocho casos"))
t3 = T3.copy(); t3.columns = ["Caso", "Dato de prueba", "Clase real"] + MODELOS
table("anexA1", "Anexo A.1: tres datos de prueba en los ocho casos (supervisado).", tabla_simple(t3, widths=[1.4, 2.4, 2.8, 2.4, 2.4, 2.4, 2.6], size=8))
t3u_ = T3u.copy(); t3u_.columns = ["Caso", "Dato de prueba", "Clase real (solo referencia)", "Cluster K-Means", "Cluster DBSCAN (−1 = ruido)"]
table("anexA2", "Anexo A.2: tres datos de prueba en los ocho casos (no supervisado).", tabla_simple(t3u_, widths=[1.4, 2.6, 4.2, 3.2, 4.2], size=8))
add(("h1u", "Anexo B. Visualizaciones por caso"))
p("El enunciado pide las visualizaciones de cada técnica para los ocho casos. En el cuerpo del documento se analiza el caso 8; aquí se incluyen todos.")
for titulo, patron, w2, cap in [("B.1 Árboles de decisión: importancia, `max_depth` vs accuracy (Gini y Entropía) y árbol Gini", "arbol_caso{c}.png", False, "Árbol de decisión"),
                                ("B.2 KNN: k vs accuracy (validación cruzada de 5 folds) y vecinos en PCA 2D", "knn_caso{c}.png", False, "KNN"),
                                ("B.3 SVM: regiones de decisión en PCA 2D", "svm_regiones_caso{c}.png", True, "SVM"),
                                ("B.4 Clusters en PCA 2D (K-Means con centroides, DBSCAN con ruido en gris)", "clusters_caso{c}.png", False, "Clusters")]:
    add(("h2u", titulo))
    items = [(patron.format(c=c), f"{cap}, caso {c}.") for c in range(1, 9)]
    if w2:
        for i in range(0, 8, 2):
            figs2(items[i:i + 2])
    else:
        for f_, c_ in items:
            fig(None, f_, c_, 15.5)

# ----------------------------------------------------------------------------------------------------------------------
# Numeración (pre-pasada): secciones, figuras y tablas; resolución de referencias
# ----------------------------------------------------------------------------------------------------------------------
sec = sub = 0; nf = nt = 0
REF = {}
NUM = []
for blk in B:
    t = blk[0]
    if t == "h1":
        sec += 1; sub = 0; NUM.append((blk[0], f"{sec}", blk[1]))
    elif t == "h2":
        sub += 1; NUM.append((blk[0], f"{sec}.{sub}", blk[1]))
    elif t == "table":
        nt += 1; REF["tab:" + blk[1]] = f"Tabla {nt}"
    elif t == "fig":
        nf += 1
        if blk[1]: REF["fig:" + blk[1]] = f"Figura {nf}"
    elif t == "figs2":
        nf += len(blk[1])
# la numeración de títulos se asigna en el mismo orden al dibujar


def resolver(s):
    return re.sub(r"\[\[(fig|tab):(\w+)\]\]", lambda m: REF.get(f"{m.group(1)}:{m.group(2)}", "¿?"), s)


# ----------------------------------------------------------------------------------------------------------------------
# Render DOCX
# ----------------------------------------------------------------------------------------------------------------------
FONT = "Times New Roman"
doc = Document()
st = doc.styles
st["Normal"].font.name = FONT; st["Normal"].font.size = Pt(11)
st["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
st["Normal"].paragraph_format.space_after = Pt(6); st["Normal"].paragraph_format.line_spacing = 1.15
for name, size, before, after in [("Heading 1", 15, 16, 6), ("Heading 2", 12.5, 12, 4)]:
    s = st[name]; s.font.name = FONT; s.font.size = Pt(size); s.font.bold = True; s.font.color.rgb = RGBColor(0, 0, 0)
    rf = s.element.rPr.rFonts
    for a_ in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a_)) is not None: del rf.attrib[qn(a_)]
    for a_ in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"): rf.set(qn(a_), FONT)
    s.paragraph_format.space_before = Pt(before); s.paragraph_format.space_after = Pt(after); s.paragraph_format.keep_with_next = True
cap = st["Caption"]; cap.font.name = FONT; cap.font.size = Pt(9.5); cap.font.bold = False; cap.font.italic = False; cap.font.color.rgb = RGBColor(0, 0, 0)
cap.paragraph_format.space_after = Pt(4); cap.paragraph_format.space_before = Pt(2)
for lst in ("List Bullet", "List Number"):
    st[lst].font.name = FONT; st[lst].font.size = Pt(11); st[lst].paragraph_format.space_after = Pt(3)

sect = doc.sections[0]
sect.page_width, sect.page_height = Cm(21.59), Cm(27.94)
for s_ in ("left_margin", "right_margin"): setattr(sect, s_, Cm(2.5))
sect.top_margin = Cm(2.5); sect.bottom_margin = Cm(2.3)
TEXT_W = 16.5
doc.core_properties.title = "Trabajo Práctico 2 – Calidad del aire en India"
doc.core_properties.author = ", ".join(INTEGRANTES)
doc.core_properties.subject = "Introducción a la Inteligencia Artificial (3010476)"


def set_cell_shade(cell, hex_):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_); tcPr.append(shd)


def set_cell_borders(cell, **kw):
    tcPr = cell._tc.get_or_add_tcPr(); b = tcPr.find(qn("w:tcBorders"))
    if b is None:
        b = OxmlElement("w:tcBorders"); tcPr.append(b)
    for edge, (val, sz) in kw.items():
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), val); e.set(qn("w:sz"), str(sz)); e.set(qn("w:color"), "000000"); b.append(e)


def set_cell_margins(table, top=30, bottom=30, left=60, right=60):
    tblPr = table._tbl.tblPr; m = OxmlElement("w:tblCellMar")
    for k, v in (("top", top), ("left", left), ("bottom", bottom), ("right", right)):
        e = OxmlElement(f"w:{k}"); e.set(qn("w:w"), str(v)); e.set(qn("w:type"), "dxa"); m.append(e)
    tblPr.append(m)


def add_hyperlink(par, url, text, size=None):
    rid = par.part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    h = OxmlElement("w:hyperlink"); h.set(qn("r:id"), rid)
    r = OxmlElement("w:r"); rPr = OxmlElement("w:rPr")
    c = OxmlElement("w:color"); c.set(qn("w:val"), "0563C1"); rPr.append(c)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rPr.append(u)
    if size:
        sz = OxmlElement("w:sz"); sz.set(qn("w:val"), str(int(size * 2))); rPr.append(sz)
    r.append(rPr); t = OxmlElement("w:t"); t.text = text; t.set(qn("xml:space"), "preserve"); r.append(t); h.append(r); par._p.append(h)


TOKEN = re.compile(r"(\*\*.+?\*\*|\*[^*\s][^*]*?\*|`[^`]+`|\^[^^]+\^|==.+?==|<<[^|>]+\|[^>]+>>)")


def runs(par, text, size=None, bold=False):
    """Escribe texto con marcado en línea dentro de un párrafo."""
    text = resolver(text)
    for tok in TOKEN.split(text):
        if not tok:
            continue
        if tok.startswith("<<"):
            url, txt = tok[2:-2].split("|", 1); add_hyperlink(par, url, txt, size); continue
        r = None
        if tok.startswith("**"): r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("==") : r = par.add_run(tok[2:-2]); r.font.highlight_color = WD_COLOR_INDEX.YELLOW; r.bold = bold
        elif tok.startswith("`"): r = par.add_run(tok[1:-1]); r.font.name = "Courier New"; r.font.size = Pt((size or 11) - 1); r.bold = bold; continue
        elif tok.startswith("^"): r = par.add_run(tok[1:-1]); r.font.superscript = True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2: r = par.add_run(tok[1:-1]); r.italic = True; r.bold = bold
        else: r = par.add_run(tok); r.bold = bold
        if size and r is not None: r.font.size = Pt(size)


def para(text, style=None, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, after=None, before=None, keep=False, bold=False, indent=None):
    par = doc.add_paragraph(style=style); par.alignment = align
    if after is not None: par.paragraph_format.space_after = Pt(after)
    if before is not None: par.paragraph_format.space_before = Pt(before)
    if keep: par.paragraph_format.keep_with_next = True
    if indent is not None: par.paragraph_format.left_indent = Cm(indent)
    runs(par, text, size, bold)
    return par


def fit_pic(run_par, path, width_cm, max_h_cm=19.5):
    from PIL import Image
    w, h = Image.open(path).size
    wcm = min(width_cm, max_h_cm * w / h)
    run_par.add_run().add_picture(path, width=Cm(wcm))


def caption(text, kind, n):
    c = doc.add_paragraph(style="Caption"); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = c.add_run(f"{kind} {n}. "); r.bold = True
    runs(c, text, 9.5)
    return c


def build_table(spec, width_cm):
    header, rows, widths, size = spec["header"], spec["rows"], spec["widths"], spec["size"]
    ncols = sum(c[1] for c in header[0])
    widths = widths or [width_cm / ncols] * ncols
    tot = sum(widths)
    if tot > width_cm + 0.01:
        widths = [w * width_cm / tot for w in widths]
    nrows = len(header) + len(rows)
    t = doc.add_table(rows=nrows, cols=ncols); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
    set_cell_margins(t)
    for ci, w in enumerate(widths):
        t.columns[ci].width = Cm(w)
        for ri in range(nrows):
            t.cell(ri, ci).width = Cm(w)
    # encabezado con celdas combinadas
    occupied = [[False] * ncols for _ in header]
    for ri, hrow in enumerate(header):
        ci = 0
        for text, cs, rs in hrow:
            while ci < ncols and occupied[ri][ci]: ci += 1
            for dr in range(rs):
                for dc in range(cs):
                    if ri + dr < len(header) and ci + dc < ncols: occupied[ri + dr][ci + dc] = True
            cell = t.cell(ri, ci)
            if cs > 1 or rs > 1: cell = cell.merge(t.cell(ri + rs - 1, ci + cs - 1))
            cell.text = ""; par = cell.paragraphs[0]; par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            par.paragraph_format.space_after = Pt(0); par.paragraph_format.line_spacing = 1.0
            runs(par, text, size, bold=True)
            set_cell_shade(cell, "D9E2F3")
            from docx.enum.table import WD_ALIGN_VERTICAL
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            ci += cs
    for ri, row in enumerate(rows):
        for ci, text in enumerate(row):
            cell = t.cell(len(header) + ri, ci); cell.text = ""; par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(0); par.paragraph_format.line_spacing = 1.0
            par.alignment = WD_ALIGN_PARAGRAPH.LEFT if (ci == 0 and spec["align_first"]) or len(text) > 28 else WD_ALIGN_PARAGRAPH.CENTER
            runs(par, text, size)
            if ri % 2 == 1: set_cell_shade(cell, "F2F5FB")
    # bordes tipo booktabs
    for ci in range(ncols):
        set_cell_borders(t.cell(0, ci), top=("single", 12))
        set_cell_borders(t.cell(len(header) - 1, ci), bottom=("single", 6))
        set_cell_borders(t.cell(nrows - 1, ci), bottom=("single", 12))
    for ri in range(nrows):
        if nrows <= 14 and ri < nrows - 1:
            for c_ in t.rows[ri].cells:
                for pp_ in c_.paragraphs: pp_.paragraph_format.keep_with_next = True
        trPr = t.rows[ri]._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if ri < len(header):
            th = OxmlElement("w:tblHeader"); trPr.append(th)
    return t


sec_n = sub_n = 0
fig_n = tab_n = 0
for blk in B:
    kind = blk[0]
    if kind == "title":
        d_ = blk[1]
        para(d_["titulo"], align=WD_ALIGN_PARAGRAPH.CENTER, size=22, after=6, bold=True).runs[0].bold = True
        para(d_["subtitulo"], align=WD_ALIGN_PARAGRAPH.CENTER, size=14, after=14, bold=True)
        para(d_["autores"], align=WD_ALIGN_PARAGRAPH.CENTER, size=12, after=8)
        for ln in d_["afil"].split("\n"):
            para(ln, align=WD_ALIGN_PARAGRAPH.CENTER, size=10.5, after=0)
        doc.add_paragraph().paragraph_format.space_after = Pt(4)
    elif kind == "abstract":
        par = para(blk[1], size=10, after=8, indent=1.0); par.paragraph_format.right_indent = Cm(1.0)
    elif kind == "h1":
        sec_n += 1; sub_n = 0
        doc.add_paragraph(f"{sec_n}  {blk[1]}", style="Heading 1")
    elif kind == "h2":
        sub_n += 1
        doc.add_paragraph(f"{sec_n}.{sub_n}  {blk[1]}", style="Heading 2")
    elif kind == "h1u":
        doc.add_paragraph(blk[1], style="Heading 1")
    elif kind == "h2u":
        h = doc.add_paragraph(style="Heading 2"); runs(h, blk[1], 12.5, bold=True)
    elif kind == "p":
        para(blk[1])
    elif kind == "note":
        par = para(blk[1], size=10.5, indent=0.6); par.paragraph_format.right_indent = Cm(0.6)
        pPr = par._p.get_or_add_pPr(); bd = OxmlElement("w:pBdr"); l = OxmlElement("w:left")
        l.set(qn("w:val"), "single"); l.set(qn("w:sz"), "18"); l.set(qn("w:space"), "6"); l.set(qn("w:color"), "8EA9DB"); bd.append(l); pPr.append(bd)
    elif kind == "eq":
        par = para(blk[1], align=WD_ALIGN_PARAGRAPH.CENTER, before=2, after=8)
        for r in par.runs: r.italic = True
    elif kind in ("ul", "ol"):
        for it in blk[1]:
            par = doc.add_paragraph(style="List Bullet" if kind == "ul" else "List Number"); par.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            runs(par, it)
    elif kind == "table":
        tab_n += 1
        caption(blk[2], "Tabla", tab_n).paragraph_format.keep_with_next = True
        width = 24.0 if doc.sections[-1].orientation == WD_ORIENT.LANDSCAPE else TEXT_W
        build_table(blk[3], width)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    elif kind == "fig":
        fig_n += 1
        par = doc.add_paragraph(); par.alignment = WD_ALIGN_PARAGRAPH.CENTER; par.paragraph_format.keep_with_next = True; par.paragraph_format.space_after = Pt(2)
        fit_pic(par, img(blk[2]), blk[4])
        caption(blk[3], "Figura", fig_n)
    elif kind == "figs2":
        t = doc.add_table(rows=2, cols=len(blk[1])); t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
        for i, (f_, c_) in enumerate(blk[1]):
            fig_n += 1
            cw = TEXT_W / len(blk[1])
            t.cell(0, i).width = Cm(cw); t.cell(1, i).width = Cm(cw)
            pp = t.cell(0, i).paragraphs[0]; pp.alignment = WD_ALIGN_PARAGRAPH.CENTER; pp.paragraph_format.space_after = Pt(0)
            fit_pic(pp, img(f_), cw - 0.3, 9.0)
            cp = t.cell(1, i).paragraphs[0]; cp.style = st["Caption"]; cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cp.add_run(f"Figura {fig_n}. "); r.bold = True; runs(cp, c_, 9.5)
        for ri in range(2):
            trPr = t.rows[ri]._tr.get_or_add_trPr(); trPr.append(OxmlElement("w:cantSplit"))
        doc.add_paragraph().paragraph_format.space_after = Pt(2)
    elif kind == "pagebreak":
        doc.add_page_break()
    elif kind == "landscape":
        new = doc.add_section(WD_SECTION.NEW_PAGE)
        if blk[1]:
            new.orientation = WD_ORIENT.LANDSCAPE; new.page_width, new.page_height = Cm(27.94), Cm(21.59)
            new.left_margin = new.right_margin = Cm(1.9); new.top_margin = Cm(2.0); new.bottom_margin = Cm(2.0)
        else:
            new.orientation = WD_ORIENT.PORTRAIT; new.page_width, new.page_height = Cm(21.59), Cm(27.94)
            new.left_margin = new.right_margin = Cm(2.5); new.top_margin = Cm(2.5); new.bottom_margin = Cm(2.3)


def campo(par, instr):
    for typ, txt in (("begin", None), (None, instr), ("end", None)):
        r = par.add_run()
        if typ:
            f = OxmlElement("w:fldChar"); f.set(qn("w:fldCharType"), typ); r._r.append(f)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; r._r.append(it)
        r.font.size = Pt(9)


for s_ in doc.sections:
    s_.header.is_linked_to_previous = False; s_.footer.is_linked_to_previous = False
    hp = s_.header.paragraphs[0]; hp.text = ""; hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run("Introducción a la Inteligencia Artificial · Trabajo Práctico 2"); r.font.size = Pt(9); r.italic = True
    fp = s_.footer.paragraphs[0]; fp.text = ""; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    campo(fp, "PAGE")

doc.save(OUT_DOCX)
print("Informe generado:", OUT_DOCX)
if AVISOS:
    print("REVISAR TEXTO: las siguientes afirmaciones del informe ya no coinciden con los datos:")
    for a in AVISOS: print("  -", a)
else:
    print("VERIFICACIÓN: todas las afirmaciones cualitativas del texto coinciden con los datos.")
if "--pdf" in sys.argv:
    exe = shutil.which("soffice") or shutil.which("libreoffice")
    if exe:
        subprocess.run([exe, "--headless", "--convert-to", "pdf", "--outdir", os.path.dirname(OUT_DOCX), OUT_DOCX], check=True, capture_output=True)
        print("PDF generado junto al .docx")
    else:
        print("LibreOffice no está instalado: exporte el PDF desde Word o Google Docs.")
shutil.rmtree(WORK, ignore_errors=True)
