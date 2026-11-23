"""Genera el documento de diseño (PDF) a partir de lo que exporta el notebook en salidas/.

Uso:  python build_informe.py [carpeta_salidas] [pdf_salida]
Los números salen de salidas/*.csv|json; el texto cualitativo se escribió con la corrida de referencia y
`verificar()` avisa por consola si alguna afirmación dejó de cumplirse tras cambiar la cédula (revisen esas frases).
Requiere: pandas, matplotlib y Chromium/Chrome (para imprimir el HTML a PDF).
"""
import glob, json, os, re, subprocess, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

D = sys.argv[1] if len(sys.argv) > 1 else "salidas"
OUT_PDF = sys.argv[2] if len(sys.argv) > 2 else "Documento_diseno.pdf"

# ======================= EDITAR ANTES DE ENTREGAR =======================
INTEGRANTES = ["Integrante 1", "Integrante 2", "Integrante 3", "Integrante 4"]
VIDEO_URL = "https://youtu.be/PENDIENTE"
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

P = {m: R.pivot(index="caso", columns="modelo", values=m) for m in ["f1", "accuracy", "precision", "recall"]}
f1 = P["f1"]
R["ED"] = R.caso.map(lambda c: SI(CASOS[c]["ED"])); R["OUT"] = R.caso.map(lambda c: SI(CASOS[c]["OUT"])); R["BAL"] = R.caso.map(lambda c: SI(CASOS[c]["BAL"]))


def efecto(factor):
    """F1 medio con el factor en SI menos NO, por modelo."""
    t = R.groupby(["modelo", factor]).f1.mean().unstack().reindex(MODELOS)
    return t["NO"], t["SI"], t["SI"] - t["NO"]


ef = {k: efecto(k) for k in ["ED", "OUT", "BAL"]}
fmt = lambda x, n=3: f"{x:.{n}f}"
sg = lambda x: "≈ 0" if abs(x) < 0.0005 else f"{x:+.3f}"

# ---------------------------- verificación de afirmaciones ----------------------------
def verificar():
    avisos = []
    if not all(np.allclose(f1.loc[i, "Árboles"], f1.loc[i + 4, "Árboles"]) for i in range(1, 5)):
        avisos.append("El árbol ya NO es idéntico con y sin escalado (casos i vs i+4).")
    if not all(ef["BAL"][2][m] > 0 for m in MODELOS):
        avisos.append("El balanceo ya no mejora el F1 medio en los 4 modelos.")
    if not (ef["ED"][2]["SVM"] > 0 and ef["ED"][2]["Redes Neuronales"] > 0):
        avisos.append("El escalado ya no mejora SVM y redes.")
    if any(abs(ef["OUT"][2][m]) > 0.03 for m in MODELOS):
        avisos.append("El efecto de los outliers superó 0.03 de F1 en algún modelo (el texto dice que es pequeño).")
    if f1["Árboles"].max() < max(f1[m].max() for m in MODELOS):
        avisos.append("El árbol ya no tiene el F1 máximo general.")
    if not (F3["K óptimo (silhouette)"] == 2).all():
        avisos.append("K-Means ya no elige K=2 en los 8 casos.")
    if not (F3.loc[F3.Balanceo == "NO", "DB clusters"] == 1).all():
        avisos.append("DBSCAN ya no da 1 cluster en los casos desbalanceados.")
    if J["roc_caso8"]["Árboles"]["auc_macro"] > min(J["roc_caso8"][m]["auc_macro"] for m in ["SVM", "Redes Neuronales"]):
        avisos.append("El árbol ya no tiene el AUC más bajo entre árbol, SVM y redes en el caso 8.")
    print("VERIFICACIÓN:", "todo coincide con el texto." if not avisos else "")
    for a in avisos:
        print("  - REVISAR TEXTO:", a)


verificar()

# ---------------------------- diagrama del pipeline ----------------------------
def diagrama():
    fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 3.6)
    cajas = [(0.1, "city_day.csv\n(29 531 filas)"), (1.85, "Subconjunto\nN según cédula\n+ semilla"), (3.6, "Split estratificado\n70 / 15 / 15"),
             (5.35, "Imputación (train)\nmediana / moda"), (7.1, "Caso 1..8:\nrecorte / outliers 5 %\nbalanceo · One-Hot\nescalado (fit train)"), (9.0, "Modelos\nÁrbol · KNN · SVM · RN\nK-Means · DBSCAN")]
    for x, t in cajas:
        ax.add_patch(FancyBboxPatch((x, 1.2), 1.6, 1.4, boxstyle="round,pad=0.05", fc="#e8f0fe", ec="#3b5bdb"))
        ax.text(x + 0.8, 1.9, t, ha="center", va="center", fontsize=8)
    for x in [1.7, 3.45, 5.2, 6.95, 8.7]:
        ax.annotate("", xy=(x + 0.2, 1.9), xytext=(x - 0.02, 1.9), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(5.5, 0.55, "Val y test reciben el mismo tratamiento en los 8 casos; encoders, escalador, imputador, SMOTE e inyección de outliers se ajustan solo con train.",
            ha="center", fontsize=8.5, style="italic")
    ax.text(5.5, 3.2, "Flujo general de preprocesamiento", ha="center", fontsize=11, weight="bold")
    plt.savefig(f"{D}/diagrama_pipeline.png", dpi=150, bbox_inches="tight"); plt.close()


diagrama()


def diagrama_red():
    """Esquema de la arquitectura del MLP (entrada -> 2 capas ocultas con L2 y Dropout -> softmax)."""
    capas = [("Entrada\n43 variables", "#e8f0fe"), ("Dense 64\nReLU · L2 1e-3", "#d3f9d8"), ("Dropout\n0.3", "#ffe3e3"),
             ("Dense 32\nReLU · L2 1e-3", "#d3f9d8"), ("Dropout\n0.3", "#ffe3e3"), ("Dense 6\nSoftmax", "#e5dbff")]
    fig, ax = plt.subplots(figsize=(11, 2.4)); ax.axis("off"); ax.set_xlim(0, 11); ax.set_ylim(0, 2.4)
    for i, (t, c) in enumerate(capas):
        x = 0.1 + i * 1.82
        ax.add_patch(FancyBboxPatch((x, 0.7), 1.4, 1.0, boxstyle="round,pad=0.04", fc=c, ec="#555"))
        ax.text(x + 0.7, 1.2, t, ha="center", va="center", fontsize=8.5)
        if i < len(capas) - 1:
            ax.annotate("", xy=(x + 1.80, 1.2), xytext=(x + 1.47, 1.2), arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(5.5, 0.25, "Adam · categorical_crossentropy (one-hot) · lotes de 64 · hasta 200 épocas · EarlyStopping (val_loss, paciencia 10)", ha="center", fontsize=8.5, style="italic")
    plt.savefig(f"{D}/diagrama_red.png", dpi=150, bbox_inches="tight"); plt.close()


diagrama_red()

# ---------------------------- helpers HTML ----------------------------
def fig(name, caption, w="100%"):
    return f'<figure style="width:{w}"><img src="{name}"><figcaption>{caption}</figcaption></figure>'


def df_html(df, cls="t", index=False):
    return df.to_html(index=index, classes=cls, border=0, escape=False)


def fig1_html():
    hdr = "".join(f'<th colspan="4">{m}</th>' for m in MODELOS)
    sub = "<th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th>" * 4
    rows = ""
    for c in range(1, 9):
        cf = CASOS[c]
        celdas = "".join(f"<td>{P[k].loc[c, m]:.3f}</td>" if k != "f1" else f"<td><b>{P[k].loc[c, m]:.3f}</b></td>"
                         for m in MODELOS for k in ["accuracy", "precision", "recall", "f1"])
        rows += f"<tr><td>{c}</td><td>CC(SI) y ED({SI(cf['ED'])})</td><td>{SI(cf['OUT'])}</td><td>{SI(cf['BAL'])}</td>{celdas}</tr>"
    return (f'<table class="t small"><tr><th rowspan="2">#</th><th rowspan="2">Normalización</th><th rowspan="2">Outliers</th><th rowspan="2">Balanceo</th>{hdr}</tr>'
            f'<tr>{sub}</tr>{rows}</table>')


def fig3_html():
    d = F3.copy()
    out = pd.DataFrame({"#": d["#"], "Normalización": d["ED"].map(lambda x: f"CC(SI) y ED({x})"), "Outliers": d["Outliers"], "Balanceo": d["Balanceo"],
                        "K-Means<br>Inertia": d["KM inertia"].map(lambda x: f"{x:,.0f}"), "K-Means<br>Silhouette": d["KM silhouette"].map(lambda x: f"{x:.3f}"),
                        "DBSCAN<br>Silhouette": d["DB silhouette"].map(lambda x: "n/a (1 cluster)" if pd.isna(x) else f"{x:.3f}"),
                        "K elegido<br>(codo)": d.apply(lambda r: f"{r['K óptimo (silhouette)']} ({r['K codo']})", axis=1),
                        "DBSCAN<br>clusters / ruido": d.apply(lambda r: f"{r['DB clusters']} / {r['DB ruido %']:.1f} %", axis=1),
                        "ε codo → usado": d.apply(lambda r: f"{r['DB eps codo']:.2f} → {r['DB eps usado']:.2f}", axis=1)})
    return df_html(out, "t small")


def rep_html():
    filas = []
    for m in MODELOS:
        r = pd.read_csv(f"{D}/reporte_clases_caso8_{m.replace(' ', '_')}.csv", index_col=0)
        filas.append([m] + [f"{r.loc[c, 'recall']:.2f} / {r.loc[c, 'f1-score']:.2f}" for c in CLASES])
    return df_html(pd.DataFrame(filas, columns=["Modelo (caso 8)"] + CLASES), "t small")


def efectos_html():
    filas = []
    for m in MODELOS:
        filas.append([m] + [f"{sg(ef[k][2][m])}" for k in ["ED", "OUT", "BAL"]])
    return df_html(pd.DataFrame(filas, columns=["Modelo", "Escalado (SI−NO)", "Outliers (SI−NO)", "Balanceo (SI−NO)"]), "t")


def ct_html(algo):
    ct = pd.read_csv(f"{D}/crosstab_caso8_{algo}.csv", index_col=0)
    ct.columns = [("ruido" if str(c) == "-1" else f"cluster {c}") for c in ct.columns]
    ct.index.name = "Clase real"
    return df_html(ct.reset_index(), "t small")


roc = J["roc_caso8"]
mx = {m: (f1[m].max(), int(f1[m].idxmax())) for m in MODELOS}
c8 = f1.loc[8]
tree_top8 = ", ".join(f"{k} ({v:.2f})" for k, v in J["arbol"]["8"]["top_features"].items())
k_bal = [J["knn_k"][str(c)] for c in range(1, 9) if CASOS[c]["BAL"]]
nn_ep = J["nn_epocas"]
casos_tr = J["casos"]
sev = {m: pd.read_csv(f"{D}/reporte_clases_caso8_{m.replace(' ', '_')}.csv", index_col=0).loc["Severe", "recall"] for m in MODELOS}
km8 = F3.iloc[7]; km1 = F3.iloc[0]; km3 = F3.iloc[2]; km5 = F3.iloc[4]; km7 = F3.iloc[6]
out_pct_km = 100 * (km3["KM inertia"] / km1["KM inertia"] - 1)
out_pct_km_ed = 100 * (km7["KM inertia"] / km5["KM inertia"] - 1)
db_unbal = F3[F3.Balanceo == "NO"]; db_bal = F3[F3.Balanceo == "SI"]
sil_ed_no = F3[F3.ED == "NO"]["KM silhouette"].mean(); sil_ed_si = F3[F3.ED == "SI"]["KM silhouette"].mean()
sil_bal_no = F3[F3.Balanceo == "NO"]["KM silhouette"].mean(); sil_bal_si = F3[F3.Balanceo == "SI"]["KM silhouette"].mean()
nul = J["nulos_pct"]
AUTORES = ", ".join(f"{n}<sup>{k + 1}</sup>" for k, n in enumerate(INTEGRANTES[:-1])) + f" y {INTEGRANTES[-1]}<sup>{len(INTEGRANTES)}</sup>"
n_train_unb = casos_tr["1"]["n_train"]; n_train_bal = casos_tr["2"]["n_train"]
ultimo_dig = (J["n_registros"] - 5000) // 150
dist = J["dist_subconjunto"]
dist_txt = ", ".join(f"{k}: {v}" for k, v in dist.items())

def anexo_html():
    t3 = T3.copy(); t3.columns = ["Caso", "Dato de prueba", "Clase real"] + MODELOS
    t3u = T3u.copy(); t3u.columns = ["Caso", "Dato de prueba", "Clase real (solo referencia)", "Cluster K-Means", "Cluster DBSCAN (-1 = ruido)"]
    h = "<h2>Anexo A. Tres datos de prueba en los 8 casos</h2><h3>A.1 Supervisado</h3>" + df_html(t3, "t small")
    h += "<h3>A.2 No supervisado</h3>" + df_html(t3u, "t small")
    h += "<h2>Anexo B. Visualizaciones por caso</h2><p>El enunciado pide las visualizaciones de cada técnica para los 8 casos. En el cuerpo del documento se analiza el caso 8; aquí van todos.</p>"
    for titulo, patron, ancho, cap in [("B.1 Árboles de decisión: importancia, max_depth vs accuracy (Gini vs Entropía) y árbol Gini", "arbol_caso{c}.png", "100%", "Árbol de decisión"),
                                      ("B.2 KNN: k vs accuracy (CV 5 folds) y vecinos en PCA 2D", "knn_caso{c}.png", "100%", "KNN"),
                                      ("B.3 SVM: regiones de decisión en PCA 2D", "svm_regiones_caso{c}.png", "49%", "SVM"),
                                      ("B.4 Clusters en PCA 2D (K-Means con centroides, DBSCAN con ruido en gris)", "clusters_caso{c}.png", "100%", "Clusters")]:
        h += f"<h3>{titulo}</h3>"
        h += '<div style="display:flex;flex-wrap:wrap;gap:6px">' if ancho != "100%" else "<div>"
        for c in range(1, 9):
            if os.path.exists(f"{D}/{patron.format(c=c)}"):
                h += fig(patron.format(c=c), f"{cap} – caso {c}", ancho)
        h += "</div>"
    return h


ANEXO = anexo_html()

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def font_faces():
    """Latin Modern (la fuente de LaTeX) desde ./fonts; si faltan, cae a Liberation Serif / Times."""
    faces = ""
    for fam, f, w, st in [("LM", "lmroman10-regular.otf", "normal", "normal"), ("LM", "lmroman10-bold.otf", "bold", "normal"),
                          ("LM", "lmroman10-italic.otf", "normal", "italic"), ("LM", "lmroman10-bolditalic.otf", "bold", "italic"),
                          ("LMMono", "lmmono10-regular.otf", "normal", "normal")]:
        path = os.path.join(FONT_DIR, f)
        if os.path.exists(path):
            faces += f"@font-face {{ font-family: '{fam}'; src: url('file://{path}'); font-weight: {w}; font-style: {st}; }}\n"
    return faces


CSS = font_faces() + """
@page { size: Letter; margin: 25mm 24mm 24mm 24mm; }
@page :left  { @top-left  { content: counter(page) "\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0 Introducción a la Inteligencia Artificial."; font: 9pt 'LM', 'Liberation Serif', serif; vertical-align: bottom; padding-bottom: 6mm; } }
@page :right { @top-right { content: "Introducción a la Inteligencia Artificial.\\00a0\\00a0\\00a0\\00a0\\00a0\\00a0 " counter(page); font: 9pt 'LM', 'Liberation Serif', serif; vertical-align: bottom; padding-bottom: 6mm; } }
@page :first { @top-right { content: none; } }
html { counter-reset: sec fig tab; }
body { font-family: 'LM', 'Latin Modern Roman', 'Liberation Serif', 'Times New Roman', serif; font-size: 10pt; line-height: 1.27; color: #000;
       text-align: justify; hyphens: auto; -webkit-hyphens: auto; }
p { margin: 0.2em 0 0.2em 0; text-indent: 1.5em; }
h2 + p, h3 + p, figure + p, table + p, .tcap + p, ul + p, ol + p, .abstract + p, div + p { text-indent: 0; }
/* título y autores */
.titleblock { text-align: center; margin: 8mm 0 7mm 0; }
.titleblock h1 { font-size: 17.3pt; font-weight: bold; margin: 0 0 14pt 0; }
.titleblock .sub { font-size: 12pt; font-weight: bold; margin-bottom: 14pt; }
.titleblock .authors { font-size: 11pt; margin-bottom: 7pt; } .titleblock sup { font-size: 7pt; }
.titleblock .aff { font-size: 9.5pt; line-height: 1.5; }
.abstract { margin: 9mm 14mm 8mm 14mm; font-size: 9.2pt; line-height: 1.3; text-indent: 0; }
/* secciones numeradas como en LaTeX */
h2 { font-size: 14.4pt; font-weight: bold; margin: 17pt 0 8pt 0; break-after: avoid; text-align: left; counter-increment: sec; counter-reset: sub; }
h2::before { content: counter(sec) "\\2003"; }
h3 { font-size: 11pt; font-weight: bold; margin: 12pt 0 5pt 0; break-after: avoid; text-align: left; counter-increment: sub; }
h3::before { content: counter(sec) "." counter(sub) "\\2003"; }
h2.unnum { counter-increment: none; counter-reset: none; } h2.unnum::before { content: none; }
h3.unnum { counter-increment: none; } h3.unnum::before { content: none; }
/* listas con guion largo */
ul { list-style: none; margin: 0.3em 0; padding-left: 2.2em; } ul li { position: relative; margin: 0.12em 0; text-align: justify; }
ul li::before { content: "\\2013"; position: absolute; left: -1.2em; }
ol { margin: 0.3em 0; padding-left: 2.2em; } ol li { margin: 0.12em 0; }
/* tablas tipo booktabs con encabezado azul claro y pie "Table N:" arriba */
.tcap { text-align: center; text-indent: 0; font-size: 9.5pt; margin: 12pt 0 3pt 0; break-after: avoid; counter-increment: tab; }
.tcap::before { content: "Table " counter(tab) ": "; }
table.t { border-collapse: collapse; width: auto; max-width: 100%; margin: 3pt auto 10pt auto; font-size: 9pt; page-break-inside: avoid;
          border-top: 1.2pt solid #000; border-bottom: 1.2pt solid #000; text-align: center; }
table.t th { background: #d4dafa; padding: 2.5pt 6pt; font-weight: bold; border: none; }
table.t td { padding: 2.2pt 6pt; border: none; text-align: center; } table.t tr:nth-child(even) td { background: #eef1fd; }
table.t td:first-child { text-align: left; }
table.small { font-size: 7.3pt; width: 100%; } table.small td, table.small th { padding: 1.6pt 2.5pt; }
/* figuras con pie "Fig. N:" abajo */
figure { margin: 8pt auto; text-align: center; page-break-inside: avoid; counter-increment: fig; } figure img { width: 100%; }
figcaption { font-size: 9pt; margin-top: 3pt; text-align: center; text-indent: 0; }
figcaption::before { content: "Fig. " counter(fig) ": "; }
.dos { display: flex; gap: 8px; align-items: flex-start; } .dos figure { width: 50%; } .dos > div { width: 50%; }
.dest, .nota { margin: 0.4em 0; text-indent: 1.5em; }
code { font-family: 'LMMono', 'Liberation Mono', monospace; font-size: 9pt; }
a { color: #000; text-decoration: none; border: 0.8pt solid #00e5e5; padding: 0 1pt; }
"""

html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Documento de diseño – Práctica 2</title><style>{CSS}</style></head><body>

<div class="titleblock">
<h1>Trabajo Práctico 2</h1>
<div class="sub">Calidad del aire en India: clasificación y agrupamiento del AQI</div>
<div class="authors">{AUTORES}</div>
<div class="aff">Universidad Nacional de Colombia, Sede Medellín, Colombia<br>Introducción a la Inteligencia Artificial (3010476) · Grupo {J['grupo']}, Equipo {J['equipo']}</div>
</div>

<div class="abstract"><b>Abstract.</b> Trabajamos con {J['n_registros']} registros de calidad del aire de 26 ciudades de India para clasificar el AQI en seis categorías (de <i>Good</i> a <i>Severe</i>) y después agrupar los datos sin etiqueta. Armamos ocho versiones del dataset combinando escalado, un 5 % de outliers y balanceo de clases, y en cada una entrenamos árbol de decisión, KNN, SVM y red neuronal; con las mismas versiones, sin la etiqueta, corrimos K-Means y DBSCAN. El F1 macro más alto fue {R.f1.max():.3f} ({R.loc[R.f1.idxmax(), 'modelo']}, caso {int(R.loc[R.f1.idxmax(), 'caso'])}). Balancear las clases fue la decisión de mayor efecto; el escalado importó para SVM y red y no para el árbol; los outliers inyectados casi no movieron los resultados. En no supervisado, K-Means separó aire limpio de contaminado y DBSCAN solo encontró estructura con los datos balanceados. Contamos también un error de preprocesamiento que nos tocó corregir a mitad de camino y que cambió por completo la lectura de los resultados.</div>

<h2>0. Lo que nos pasó de entrada</h2>
<p>Lo primero que nos pegó fue una sorpresa fea. Con el pipeline recién armado, la red neuronal sacaba un F1 de 0.15 en el caso 7 y la SVM se quedaba en 0.43 en el caso 8, mientras el árbol de decisión seguía tranquilo en 0.52 y 0.65. Nos tocó destapar el código para enredarnos con el problema, y no estaba en los modelos: para “limpiar” los outliers recortábamos solo el train, y el test seguía con sus valores crudos. Después de estandarizar, algunas filas de prueba quedaban hasta 98 desviaciones por fuera de lo que el modelo había visto, y en el caso 8 el 40 % de las filas de prueba tenía al menos un valor con |z| &gt; 10. KNN, SVM y redes no fallaban por ser malos, fallaban por un test que no se parecía a su entrenamiento.</p>
<p>Lo corregimos aplicando a validación y prueba el mismo recorte (con límites aprendidos solo en train) en los 8 casos. De rebote quedó algo que el enunciado pide: el test es idéntico en valores y en registros en todos los casos, así que la figura 1 se puede comparar fila contra fila. Todo lo que sigue sale de esa versión corregida.</p>
<div class="dest"><b>Lo más importante que vimos:</b> (1) balancear las clases movió el F1 más que cualquier otra decisión (entre {sg(min(ef['BAL'][2]))} y {sg(max(ef['BAL'][2]))} de F1 medio según el modelo); (2) el escalado cambia mucho a SVM y redes y nada al árbol; (3) los outliers que inyectamos movieron el F1 menos de lo que esperábamos (alrededor de ±0.01); (4) en no supervisado, K-Means separa aire “limpio” de aire “contaminado” y DBSCAN solo encuentra estructura cuando el balanceo le densifica los datos.</div>

<h2>1. Dataset: fuente, variables y justificación</h2>
<h3>1.1 Fuente y por qué lo escogimos</h3>
<p>Usamos <i>Air Quality Data in India</i> (Kaggle, Rohan Rao, <a href="{DATASET_URL}">enlace</a>), archivo <code>city_day.csv</code>: mediciones diarias de contaminantes en 26 ciudades de India entre 2015 y 2020. Lo elegimos porque (a) la variable objetivo, <code>AQI_Bucket</code>, trae 6 clases ordenadas por severidad; (b) mezcla variables numéricas y categóricas; (c) tiene nulos y valores extremos de verdad, así que el preprocesamiento que pide el enunciado importa en la práctica y no queda de adorno; y (d) los errores tienen costo claro: no es lo mismo decirle a una ciudad que el aire está “Moderate” cuando está “Severe” que al revés.</p>
<p>Las categorías siguen los rangos del índice de calidad del aire de India (CPCB), que es lo que hay detrás de <code>AQI_Bucket</code>:</p>
<table class="t"><tr><th>AQI_Bucket</th><th>Good</th><th>Satisfactory</th><th>Moderate</th><th>Poor</th><th>Very Poor</th><th>Severe</th></tr>
<tr><td>Rango de AQI</td><td>0–50</td><td>51–100</td><td>101–200</td><td>201–300</td><td>301–400</td><td>401–500</td></tr></table>
<h3>1.2 Descripción de las variables</h3>
<table class="t">
<tr><th>Variable</th><th>Tipo</th><th>Uso en el modelo</th></tr>
<tr><td>City</td><td>Categórica nominal (26 ciudades)</td><td>One-Hot</td></tr>
<tr><td>Date</td><td>Fecha (2015–2020)</td><td>Se derivan <b>Season</b> (Winter, Summer, Monsoon, PostMonsoon; nominal → One-Hot) y <b>Year</b> (ordinal por orden temporal)</td></tr>
<tr><td>PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene</td><td>Numéricas continuas</td><td>Imputación con mediana, recorte IQR, escalado</td></tr>
<tr><td>AQI</td><td>Numérica</td><td><b>Se excluye</b>: <code>AQI_Bucket</code> se calcula directamente a partir de ella (fuga de información)</td></tr>
<tr><td>AQI_Bucket</td><td>Objetivo, 6 clases</td><td>Good &lt; Satisfactory &lt; Moderate &lt; Poor &lt; Very Poor &lt; Severe</td></tr>
</table>
<h3>1.3 Verificación de requisitos</h3>
<ul>
<li>Cédula utilizada: <b>{J['cedula']}</b> → último dígito distinto de cero = {ultimo_dig} → 5000 + {ultimo_dig} × 150 = <b>{J['n_registros']}</b> registros; semilla <b>{J['seed']}</b> (dos últimos dígitos distintos de cero).</li>
<li>Registros originales: <b>{J['filas_originales']:,}</b>; con etiqueta: <b>{J['filas_etiquetadas']:,}</b>. Subconjunto de trabajo: <b>{J['n_registros']}</b> (≥ 5000 ✔), muestreado al azar con <code>random_state={J['seed']}</code> (distinto de 42 ✔).</li>
<li>Clases en la variable objetivo: <b>6</b> (≥ 4 ✔). Distribución del subconjunto: {dist_txt}.</li>
<li>Características numéricas (12) y categóricas (City, Season, Year) ✔.</li>
<li>Partición estratificada: train {J['tamanos_split']['tr']}, validación {J['tamanos_split']['va']}, prueba {J['tamanos_split']['te']} (70/15/15).</li>
</ul>

<h2>2. Preprocesamiento</h2>
{fig('diagrama_pipeline.png', 'Figura A. Flujo del preprocesamiento. Todo lo que “aprende” algo de los datos se ajusta solo con train.')}
<h3>2.1 Análisis exploratorio</h3>
<p>Nulos: Xylene es la peor ({nul['Xylene']:.1f} %), seguida de PM10 ({nul['PM10']:.1f} %), NH3 ({nul['NH3']:.1f} %) y Toluene ({nul['Toluene']:.1f} %); el resto está por debajo del 15 %. Imputamos con la <b>mediana</b> (calculada en train) porque los contaminantes tienen colas muy pesadas y la media se la comen los picos. Con Xylene la imputación es discutible: más de la mitad de la columna es mediana. La dejamos para respetar el enunciado, pero es una limitación que anotamos al final.</p>
<p>Correlaciones altas (|r| &gt; 0.8): {', '.join(f'{k} ({v})' for k, v in J['correlaciones_altas'].items())}. Es esperable: PM2.5 está contenido en PM10, y NOx incluye al NO. Para árboles no estorba; para KNN y SVM significa que esa información pesa doble en la distancia.</p>
<div class="dos">{fig('eda_clases.png', 'Figura B. Distribución de clases del subconjunto.')}{fig('eda_correlacion.png', 'Figura C. Matriz de correlación.')}</div>
<h3>2.2 Decisiones y justificación</h3>
<ul>
<li><b>Conversión de categóricas (CC):</b> One-Hot para <code>City</code> y <code>Season</code> porque no tienen un orden natural (Medellín no es “mayor” que Bogotá, ni el monzón mayor que el invierno). Codificación ordinal para <code>Year</code>, que sí tiene orden temporal. Esta es la decisión más discutible del dataset: casi no trae ordinales de verdad y <code>Year</code> fue la opción menos forzada.</li>
<li><b>Escalado (ED):</b> <code>StandardScaler</code>, con fit solo en train. Lo preferimos a MinMax porque con colas pesadas un solo valor extremo comprime a todos los demás en un rango minúsculo.</li>
<li><b>Outliers:</b> detección con IQR (factor 1.5) sobre las 12 variables numéricas, con límites calculados en train. Con ese criterio el <b>{J['outliers_train_natural_filas_pct']} %</b> de las filas de train tiene al menos un valor atípico (Xylene {J['outliers_por_variable_pct']['Xylene']} %, Toluene {J['outliers_por_variable_pct']['Toluene']} %), así que <i>eliminarlos</i> habría botado más de la mitad de los datos. Por eso <b>“sin outliers” = recortar (winsorizing) a los límites IQR</b>. <b>“Con outliers”</b> = al train ya recortado le inyectamos sintéticamente exactamente el 5 % de filas con un valor por encima del límite superior en una variable numérica al azar (entre 1.5 y 4 IQR por encima): {casos_tr['3']['outliers']['n_inyectados']} filas en los casos 3 y 7 (de {casos_tr['3']['n_train']}) y {casos_tr['4']['outliers']['n_inyectados']} en los casos 4 y 8 (de {casos_tr['4']['n_train']}). Validación y prueba reciben siempre el recorte con los límites de train.</li>
<li><b>Desbalanceado:</b> submuestreo del train hasta que la clase mayoritaria (Moderate) sea ≈70 % y las otras cinco queden iguales ({casos_tr['1']['dist_train']}). Quedan {n_train_unb} filas de train.</li>
<li><b>Balanceado:</b> SMOTE sobre el train natural, hasta igualar las 6 clases a {max(casos_tr['2']['dist_train'].values())} cada una ({n_train_bal} filas). Elegimos SMOTE porque interpola entre vecinos reales en vez de repetir filas, y se aplica en espacio estandarizado temporal para que las distancias no las mande CO o PM10 por tener unidades grandes. Las dummies se redondean a 0/1 después.</li>
<li><b>Reproducibilidad:</b> el mismo <code>random_state={J['seed']}</code> en la muestra, la partición, los modelos, SMOTE, la inyección de outliers y las inicializaciones de la red. Corrimos el notebook completo dos veces y obtuvimos exactamente las mismas métricas.</li>
</ul>
<div class="nota"><b>Cosa a tener en cuenta:</b> “desbalanceado” y “balanceado” no solo cambian la proporción de clases, también el tamaño del train ({n_train_unb} vs {n_train_bal} filas). Parte de la mejora que atribuimos al balanceo puede venir de tener más filas (sintéticas) para entrenar.</div>

<h2>3. Aprendizaje supervisado</h2>
<h3>3.1 Modelos y configuración</h3>
<table class="t">
<tr><th>Técnica</th><th>Configuración</th></tr>
<tr><td>Árbol de decisión</td><td>Se probaron Gini y Entropía con max_depth ∈ {{2,3,4,5,6,8,10,12,15,20}}; se elige la mejor combinación por F1 macro en validación. En los 8 casos ganó Gini (profundidad 5 en la mayoría; 10 en los casos balanceados sin outliers, 2 y 6).</td></tr>
<tr><td>KNN</td><td>k impar entre 1 y 25 elegido por accuracy medio de validación cruzada de 5 folds (k elegido por caso: {', '.join(str(J['knn_k'][str(c)]) for c in range(1, 9))}).</td></tr>
<tr><td>SVM</td><td>Kernel RBF, C = 1, gamma = “scale”. Para las curvas ROC/PR se usan los puntajes de <code>decision_function</code> (no son probabilidades calibradas).</td></tr>
<tr><td>Red neuronal (MLP)</td><td>Dense(64, ReLU, L2 1e-3) → Dropout 0.3 → Dense(32, ReLU, L2 1e-3) → Dropout 0.3 → Dense(6, softmax). Adam, pérdida <code>categorical_crossentropy</code> con etiquetas one-hot (<code>to_categorical</code>), lotes de 64, hasta 200 épocas, con el conjunto de validación en <code>validation_data</code>. Regularización y callback como en el taller 8 de clase: L2, Dropout y <code>EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)</code>. Paró entre {min(nn_ep.values())} y {max(nn_ep.values())} épocas según el caso.</td></tr>
</table>
{fig('diagrama_red.png', 'Figura Q. Arquitectura de la red neuronal (MLP) usada en los 8 casos.')}
<p>Configurar la red fue cacharrear: arrancamos con una arquitectura pequeña y le fuimos sumando regularización hasta que las curvas de train y validación dejaron de separarse. Métricas: accuracy y precision/recall/F1 <b>macro</b> (promedio simple por clase), porque queremos que Severe, que es una clase pequeña, pese lo mismo que Moderate.</p>

<h3>3.2 Resultados (Figura 1 del enunciado)</h3>
{fig1_html()}
<p style="font-size:8.5pt;color:#555">ED = escalado; CC siempre aplicada. Métricas sobre el mismo conjunto de prueba ({J['tamanos_split']['te']} registros) en los 8 casos. En negrita, el F1 macro.</p>
<h3>3.3 Gráficas pedidas</h3>
<div class="dos">{fig('f1_max_algoritmo.png', 'Figura D. F1 máximo por algoritmo.')}{fig('f1_medio_ed.png', 'Figura E. F1 medio por escalado y algoritmo.')}</div>
<div class="dos">{fig('f1_medio_outliers.png', 'Figura F. F1 medio por manejo de outliers y algoritmo.')}{fig('f1_medio_balanceo.png', 'Figura G. F1 medio por balanceo y algoritmo.')}</div>
<p><b>Efecto medio de cada decisión sobre el F1 (promedio de los 4 casos con el factor en SI menos el promedio de los 4 en NO):</b></p>
{efectos_html()}

<h3>Patrones generales</h3>
<ul>
<li><b>Balanceo:</b> es la decisión que más movió el F1 en los cuatro modelos ({', '.join(f"{m} {sg(ef['BAL'][2][m])}" for m in MODELOS)}).</li>
<li><b>Escalado:</b> decisivo para SVM ({sg(ef['ED'][2]['SVM'])}) y red ({sg(ef['ED'][2]['Redes Neuronales'])}), irrelevante para el árbol ({sg(ef['ED'][2]['Árboles'])}) y mixto para KNN ({sg(ef['ED'][2]['KNN'])}).</li>
<li><b>Outliers:</b> efecto pequeño en todos ({sg(min(ef['OUT'][2]))} a {sg(max(ef['OUT'][2]))}).</li>
<li><b>Ranking por F1 máximo:</b> {' > '.join(f"{m} ({v[0]:.3f})" for m, v in sorted(mx.items(), key=lambda kv: -kv[1][0]))}.</li>
</ul>

<h3>3.4 Análisis de resultados</h3>
<p><b>Quién gana.</b> El árbol tiene el F1 máximo general ({fmt(mx['Árboles'][0])}, caso {mx['Árboles'][1]}), seguido de SVM ({fmt(mx['SVM'][0])}, caso {mx['SVM'][1]}), redes ({fmt(mx['Redes Neuronales'][0])}, caso {mx['Redes Neuronales'][1]}) y KNN ({fmt(mx['KNN'][0])}, caso {mx['KNN'][1]}). Pero la ventaja del árbol es corta y se desvanece en el caso 8 (todo el preprocesamiento aplicado): árbol {fmt(c8['Árboles'])}, SVM {fmt(c8['SVM'])}, red {fmt(c8['Redes Neuronales'])}, KNN {fmt(c8['KNN'])}. El árbol gana más que todo cuando los demás están mal preparados, y esa fue la lección del escalado.</p>
<p><b>Sensibilidad a la escala.</b> El árbol dio <i>exactamente</i> el mismo resultado con y sin escalado (casos 1=5, 2=6, 3=7, 4=8): divide por umbrales de una variable a la vez y escalar solo mueve el umbral de sitio. SVM y red sí se benefician ({sg(ef['ED'][2]['SVM'])} y {sg(ef['ED'][2]['Redes Neuronales'])} de F1 medio): con unidades crudas, PM10 (cientos) y CO (unidades) no pesan igual en el kernel RBF ni en los pesos de la red. KNN fue la excepción ({sg(ef['ED'][2]['KNN'])}): escalar le ayuda en los casos balanceados y le estorba en los desbalanceados, y no encontramos una explicación que podamos comprobar, así que no la damos por hecha.</p>
<p><b>Outliers.</b> El efecto fue pequeño en los 4 modelos (entre {sg(min(ef['OUT'][2]))} y {sg(max(ef['OUT'][2]))}). Nos esperábamos más, sobre todo en SVM, KNN y redes. Creemos que pesan dos cosas: inyectamos un 5 % moderado (1.5 a 4 IQR por encima del límite) y el test va recortado, de modo que los modelos nunca ven en evaluación los valores que se inyectaron en entrenamiento. Es decir, medimos cuánto estorba entrenar con ruido, no cuánto estorba recibir ruido al predecir. Esto último sí lo vivimos sin querer en la sección 1 y fue mucho más destructivo.</p>
<p><b>Desbalance y por qué el accuracy engaña.</b> Balancear fue la decisión con más efecto ({sg(ef['BAL'][2]['Árboles'])} en árbol, {sg(ef['BAL'][2]['KNN'])} en KNN, {sg(ef['BAL'][2]['SVM'])} en SVM y {sg(ef['BAL'][2]['Redes Neuronales'])} en redes). Con el train dominado por Moderate (70 %), los modelos aprenden que acertar “Moderate” casi siempre paga, y las clases chicas (Good, Severe) quedan mal atendidas. Un clasificador tonto que dijera siempre “Moderate” tendría 70 % de accuracy dentro de ese train sin haber aprendido nada. Aquí hay que ser honestos: como el test conserva la distribución natural (Moderate es ~36 %), el accuracy no se infló tanto como pasaría con un test 70/30; el engaño se ve mejor mirando el recall por clase que el accuracy global. Por eso reportamos F1 macro y no accuracy como métrica principal.</p>
<p><b>Costo de los errores y métrica que priorizamos.</b> En calidad del aire un falso negativo (decir “Moderate” cuando el aire está en “Severe”) deja a la gente sin alerta; un falso positivo (alertar de más) genera molestia y costo, pero no daño directo a la salud. Por eso, entre las métricas, miramos primero el <b>recall de las clases altas (Poor, Very Poor, Severe)</b> y usamos <b>F1 macro</b> como resumen, porque no deja que Moderate y Satisfactory tapen los fallos en las clases raras.</p>
{rep_html()}
<p style="font-size:8.5pt;color:#555">Recall / F1 por clase, caso 8, conjunto de prueba. El recall de Severe fue {', '.join(f'{m} {sev[m]:.2f}' for m in MODELOS)}.</p>
<h3>3.5 Matrices de confusión y curvas ROC / Precision-Recall (caso 8)</h3>
{fig('matrices_confusion_caso8.png', 'Figura H. Matrices de confusión de las 4 técnicas en el caso 8.')}
{fig('roc_pr_caso8.png', 'Figura I. Curvas ROC y Precision-Recall (promedio micro) del caso 8.')}
<p>El AUC macro (uno contra todos) del caso 8 fue: red {roc['Redes Neuronales']['auc_macro']:.3f}, SVM {roc['SVM']['auc_macro']:.3f}, árbol {roc['Árboles']['auc_macro']:.3f} y KNN {roc['KNN']['auc_macro']:.3f}; el AUC-PR micro: red {roc['Redes Neuronales']['auc_pr_micro']:.3f}, SVM {roc['SVM']['auc_pr_micro']:.3f}, árbol {roc['Árboles']['auc_pr_micro']:.3f}, KNN {roc['KNN']['auc_pr_micro']:.3f}. Aunque el árbol empata en F1 con SVM y red, su curva es peor: un árbol de profundidad 5 solo reparte unas pocas probabilidades distintas (cada hoja da un valor), así que la curva sale escalonada y pierde resolución para ordenar casos. KNN con k=1 es el caso extremo: sus “probabilidades” son 0 o 1 y por eso tiene el AUC ROC más bajo ({roc['KNN']['auc_macro']:.3f}). Su AUC-PR ({roc['KNN']['auc_pr_micro']:.3f}) sale más alta que la del árbol, pero no la leemos como mejor: con tan pocos umbrales la curva tiene muy pocos puntos y el área por trapecio (el cálculo que vimos en el taller 7) la favorece, así que para KNN nos fiamos más del AUC ROC y del F1. SVM y red ordenan mejor los casos. Ojo: los puntajes de SVM no son probabilidades calibradas, por eso comparamos las curvas como ranking y no como probabilidad.</p>

<h3>3.6 Visualización por técnica</h3>
<p><b>Árbol de decisión.</b> Las variables más importantes en el caso 8 son: {tree_top8}.</p>
{fig('arbol_caso8.png', 'Figura J. Árbol (caso 8): importancia de características, max_depth vs accuracy (Gini vs Entropía) y primeros niveles del árbol con Gini. Los 8 casos están en salidas/arbol_casoN.png y las reglas en salidas/reglas_arbol_casoN.txt.')}
<p><b>Lectura en el dominio.</b> Que PM2.5 y PM10 manden tiene sentido: el AQI indio se calcula a partir de los subíndices de cada contaminante y el material particulado suele ser el que más pesa. La regla raíz parte por PM10 por debajo o por encima del promedio; con PM10 muy bajo y CO bajo el árbol cae en <i>Good</i>; con PM2.5 alto (&gt; 1.5 desviaciones) cae en <i>Very Poor</i>/<i>Severe</i>. Un hallazgo menos cómodo: aparece <code>City_Ahmedabad</code> como la tercera variable más importante. El árbol usa esa ciudad como atajo para desempatar, porque Ahmedabad tiene niveles persistentemente altos. Es útil para predecir, pero es información de lugar y no de contaminación, y no deberíamos creer que “estar en Ahmedabad” causa un AQI peor.</p>
<p><b>Subajuste y sobreajuste.</b> La exactitud de entrenamiento sube hasta ~1.0 al crecer max_depth, mientras la de validación se aplana pasada la profundidad 5; ahí está el sobreajuste. Con profundidad 2–3 hay subajuste (ambas bajas). Gini y Entropía dieron curvas casi idénticas y Gini ganó en los 8 casos, así que usamos Gini.</p>
<p><b>KNN.</b> En los casos balanceados la validación cruzada eligió k = {', '.join(map(str, k_bal))} (siempre 1) y con k=1 daba un accuracy de ~0.86, pero en prueba el accuracy fue ~0.68. Es un espejismo de SMOTE: los puntos sintéticos son interpolaciones de puntos reales, y al repartirse entre folds, un punto de validación tiene casi un gemelo en entrenamiento. Es una limitación que dejamos escrita: para los casos balanceados el k “elegido” sale optimista.</p>
{fig('knn_caso8.png', 'Figura K. KNN (caso 8): k vs accuracy (CV 5 folds) y vecinos de 3 puntos de prueba en PCA 2D (los vecinos se calculan en el espacio completo; PCA solo dibuja).')}
<p><b>SVM.</b> Las regiones son franjas curvas que se ordenan de izquierda a derecha (aire más limpio → más contaminado), y se solapan bastante en el centro, que es donde viven Moderate y Poor. Con dos componentes se pierde información, así que el solapamiento del dibujo exagera el del modelo real.</p>
{fig('svm_regiones_caso8.png', 'Figura L. Regiones de decisión de la SVM en PCA 2D (caso 8). Se reentrena una SVM sobre la proyección solo para poder dibujar; el modelo evaluado usa las variables completas.', '70%')}
<p><b>Red neuronal.</b> En el caso 8 la validación se aplana temprano y EarlyStopping paró en la época {nn_ep['8']} restaurando los mejores pesos; la brecha entre train y validación es moderada gracias al Dropout y la regularización L2.</p>
{fig('nn_curvas_caso8.png', 'Figura M. Curvas de pérdida y accuracy de la red (caso 8).')}

<h3>3.7 Tres datos de prueba (caso 8)</h3>
{df_html(T3[T3.caso == 8].drop(columns="caso"))}
<p style="font-size:8.5pt;color:#555">Los mismos 3 registros de prueba en los 8 casos (todos los casos están en <code>salidas/tres_datos_prueba_supervisado.csv</code>). Tres ejemplos sirven para ilustrar, no para concluir nada: la evidencia es la tabla de la figura 1.</p>

<h3>3.8 Conclusiones del módulo supervisado</h3>
<ul>
<li>Con buen preprocesamiento (caso 8) árbol, SVM y red quedan parecidos en F1 (≈{fmt(min(c8[['Árboles','SVM','Redes Neuronales']]))}–{fmt(max(c8[['Árboles','SVM','Redes Neuronales']]))}); KNN queda un poco atrás. La diferencia grande no está entre algoritmos sino entre prepararlos bien o mal.</li>
<li>El árbol es el más robusto al preprocesamiento (no depende del escalado y casi no reacciona a los outliers) y el más fácil de explicar. SVM y red necesitan escalado y rinden mejor ordenando casos (ROC/PR).</li>
<li>Para este dominio, nos quedaríamos con un SVM o red entrenados sobre datos escalados y balanceados si importa el ranking de riesgo, y con el árbol si importa explicarle la regla a alguien. En ambos casos vigilaríamos el recall de Severe y no el accuracy.</li>
</ul>

<h2>4. Aprendizaje no supervisado</h2>
<h3>4.1 Construcción de los casos</h3>
<p>Usamos el mismo dataset sin la variable objetivo. Las 8 versiones de la figura 2 son exactamente los mismos train del módulo supervisado (mismos registros, mismo recorte/inyección de outliers, mismo balanceo, One-Hot y escalado) con la etiqueta descartada. Así, “balanceado” significa que los datos incluyen los puntos sintéticos de SMOTE, y “desbalanceado”, el submuestreo con 70 % de Moderate. La etiqueta original se usó únicamente para la validación externa (cruzar cada cluster con la clase real) y para interpretar los clusters, nunca para entrenar.</p>
<h3>4.2 Configuración</h3>
<ul>
<li><b>K-Means:</b> inicialización k-means++, <code>n_init=10</code>, <code>random_state={J['seed']}</code>, K de 2 a 10. Se reportan inertia (codo) y silhouette (calculado con muestra de 3000 puntos para que no se demore); el K elegido es el de mayor silhouette, y el K del codo se muestra al lado.</li>
<li><b>DBSCAN:</b> MinPts = 2 × número de características (= {2 * casos_tr['1']['n_features']}, con {casos_tr['1']['n_features']} columnas tras el One-Hot). ε se toma del codo de la curva de k-distancia (k = MinPts). Silhouette sin contar el ruido (necesita ≥ 2 clusters).</li>
</ul>
<div class="nota"><b>Ajuste que tuvimos que hacer:</b> con el ε del codo, DBSCAN dejaba todos los datos en <b>un solo cluster</b> en varios casos (silhouette no calculable). Cuando pasó, exploramos 12 valores de ε entre 0.3 y 0.95 veces el del codo y nos quedamos con el de mayor silhouette que diera ≥ 2 clusters y ≤ 50 % de ruido. La tabla muestra ambos ε. Si ningún ε cumplía, dejamos el del codo, y en esos casos no hay silhouette.</div>
<h3>4.3 Resultados (Figura 3 del enunciado)</h3>
{fig3_html()}
{fig('silhouette_por_caso.png', 'Figura N. Silhouette por caso para K-Means (izquierda) y DBSCAN (derecha; 0 = no calculable).')}
{fig('kmeans_dbscan_seleccion_parametros.png', 'Figura O. Selección de parámetros por caso: codo (inertia), silhouette vs K y gráfico de k-distancia (línea roja: ε del codo; verde punteada: ε usado).', '72%')}
<h3>4.4 Análisis</h3>
<p><b>K-Means.</b> El silhouette eligió K=2 en los 8 casos, mientras el codo sugirió entre {int(F3['K codo'].min())} y {int(F3['K codo'].max())}. Esa diferencia dice algo: las 6 clases del AQI son más un continuo (cada clase es un tramo de la misma escala) que 6 grupos separados, así que el único corte limpio es “aire más limpio vs más contaminado”. Cruzando con la etiqueta (caso 8), el cluster 1 junta casi todo Good, Satisfactory y la mayor parte de Moderate, y el cluster 0 junta la mayoría de Poor, Very Poor y Severe. Sí tiene sentido para el dominio, aunque con K=2 no se pueden recuperar las 6 clases.</p>
{ct_html('kmeans')}
{ct_html('dbscan')}
<p style="font-size:8.5pt;color:#555">Cruces de las Tablas anteriores (caso 8). Solo para interpretar; la etiqueta no entró al entrenamiento.</p>
<p><b>Efecto del escalado.</b> La inertia no se puede comparar entre escalado y no escalado: sin escalar es del orden de millones ({km1['KM inertia']:,.0f} en el caso 1) y escalada de decenas de miles ({km5['KM inertia']:,.0f} en el caso 5), simplemente por las unidades. El silhouette sin escalar fue más alto en promedio ({sil_ed_no:.3f} vs {sil_ed_si:.3f}), pero no lo leemos como “mejor”: sin escalar, las variables de magnitud grande (PM10, NOx) mandan en la distancia euclidiana y K-Means termina partiendo por casi una sola variable, lo cual hace fácil que el silhouette salga alto.</p>
<p><b>Efecto de los outliers.</b> Inyectar el 5 % subió la inertia ~{out_pct_km:.0f} % sin escalar (caso 1→3) y ~{out_pct_km_ed:.0f} % con escalado (caso 5→7), mientras el silhouette casi no se movió. Es lo esperado: la inertia suma distancias al cuadrado y unos pocos puntos lejanos la inflan, pero el centroide se corre poco con solo 5 %.</p>
<p><b>Efecto del balanceo.</b> El silhouette promedio subió de {sil_bal_no:.3f} a {sil_bal_si:.3f} con los datos balanceados. Ojo con leerlo como una mejora real: SMOTE rellena el espacio entre puntos reales con puntos interpolados, lo que hace los grupos más compactos y el silhouette se infla.</p>
<p><b>DBSCAN.</b> En los 4 casos desbalanceados dio <b>un solo cluster</b> (ruido ≤ {db_unbal['DB ruido %'].max():.1f} %): con {n_train_unb} puntos en {casos_tr['1']['n_features']} dimensiones y MinPts = {2 * casos_tr['1']['n_features']}, la densidad es tan baja que hace falta un ε grande para que haya núcleos, y a ese ε todo se conecta. En los casos balanceados, que tienen cuatro veces más puntos, sí aparecieron estructuras: {int(db_bal['DB clusters'].min())}–{int(db_bal['DB clusters'].max())} clusters, con {db_bal['DB ruido %'].min():.0f}–{db_bal['DB ruido %'].max():.0f} % de ruido. Con escalado los clusters fueron más ({int(F3.loc[(F3.ED=='SI') & (F3.Balanceo=='SI'), 'DB clusters'].max())}) y más ruido. El cruce con la etiqueta del caso 8 muestra qué encontró: un cluster grande de aire limpio (Good y Satisfactory), bolsillos densos dominados por Severe, y el ruido concentra Moderate y Poor, la zona intermedia donde no hay vacíos de densidad. Esto confirma lo que sospechábamos: DBSCAN encuentra los extremos, pero la franja del medio es un continuo y no tiene “bordes” que detectar.</p>
<h3>K-Means vs DBSCAN</h3>
<table class="t">
<tr><th></th><th>K-Means</th><th>DBSCAN</th></tr>
<tr><td>Forma de los clusters</td><td>Esféricos/convexos (distancia al centroide)</td><td>Arbitraria; en el PCA se ven manchas alargadas</td></tr>
<tr><td>Definir K</td><td>Hay que fijarlo (codo/silhouette); aquí dio 2</td><td>No se define K, pero hay que acertar con ε y MinPts (más delicado que K)</td></tr>
<tr><td>Ruido y outliers</td><td>Todo punto cae en algún cluster; los outliers arrastran la inertia</td><td>Marca el ruido explícitamente; en balanceados fue 15–43 %</td></tr>
<tr><td>Densidades variables</td><td>Indiferente a la densidad, solo mira distancia</td><td>Sufre: un solo ε no sirve para zonas densas y dispersas a la vez (aquí, 1 cluster o mucho ruido)</td></tr>
<tr><td>Alta dimensión ({casos_tr['1']['n_features']} columnas)</td><td>Aguanta mejor</td><td>La distancia pierde contraste y la densidad se diluye</td></tr>
</table>
<h3>4.5 Visualización de clusters en 2D (PCA)</h3>
{fig('clusters_caso8.png', 'Figura P. Caso 8: K-Means (K elegido, centroides en rojo) y DBSCAN (ruido en gris). Los 8 casos están en salidas/clusters_casoN.png.')}
<p>Se ve lo que cuentan las tablas: K-Means corta la nube en dos mitades por el eje principal (PC1, que básicamente ordena de aire más limpio a más contaminado), y DBSCAN recoge las zonas densas y deja en gris la parte intermedia.</p>
<h3>4.6 Tres datos de prueba (caso 8)</h3>
{df_html(T3u[T3u.caso == 8].drop(columns="caso"))}
<p style="font-size:8.5pt;color:#555">K-Means asigna al centroide más cercano; DBSCAN no tiene <code>predict</code>, así que usamos el punto núcleo más cercano a distancia ≤ ε, o ruido si queda más lejos.</p>
<h3>4.7 Conclusiones del módulo no supervisado</h3>
<ul>
<li>Los datos no traen 6 grupos naturales: el AQI es una escala continua dividida por umbrales. El agrupamiento recupera bien el eje “limpio ↔ contaminado” y mal las clases intermedias.</li>
<li>K-Means es estable y fácil de interpretar pero obliga a partir todo; DBSCAN es más honesto con el ruido pero muy sensible a ε, MinPts y al tamaño del conjunto.</li>
<li>Las comparaciones entre casos con y sin escalado o con y sin balanceo hay que leerlas con cuidado: la inertia depende de las unidades y SMOTE infla la cohesión.</li>
</ul>

<h2>5. Conclusiones generales y limitaciones</h2>
<ul>
<li>Lo que más movió los resultados fue lo menos vistoso: cómo prepararon los datos (balanceo y escalado) y que el test se tratara igual que el train. Ningún algoritmo ganó por sí solo.</li>
<li>El mismo patrón se repite en las dos mitades: el árbol y K-Means son robustos y explicables; SVM, red y DBSCAN rinden o fallan según el preprocesamiento.</li>
<li><b>Limitaciones.</b> (1) Una sola corrida por caso, sin repeticiones con varias semillas: las diferencias de ~0.01 no son concluyentes. (2) El tamaño de train cambia entre balanceado y desbalanceado. (3) Xylene tiene {nul['Xylene']:.0f} % de nulos imputados con mediana. (4) El k de KNN en casos balanceados sale optimista por SMOTE. (5) <code>Year</code> como variable ordinal es una decisión discutible. (6) El test va recortado con límites de train, así que no mide el desempeño ante outliers reales. (7) El ajuste de ε en DBSCAN es una decisión nuestra, descrita en 5.2.</li>
</ul>

<h2>6. Qué sale de las clases y qué viene del enunciado</h2>
<p>Para la sustentación nos sirvió dejar claro de dónde sale cada herramienta. Todo lo que está en la columna de clase aparece en las diapositivas o en los talleres.</p>
<table class="t">
<tr><th>Tema</th><th>Qué usamos</th><th>Dónde se vio</th></tr>
<tr><td>Partición de datos</td><td>Train 70 % / validación 15 % / prueba 15 %; la validación decide hiperparámetros y el test se toca al final</td><td>Clase de redes neuronales: conjuntos de entrenamiento, validación y prueba</td></tr>
<tr><td>Escalado</td><td>StandardScaler (ajustado en train)</td><td>Taller 8 (StandardScaler); clase de K-Means (Min-Max)</td></tr>
<tr><td>Árboles</td><td><code>DecisionTreeClassifier</code>, Gini vs Entropía, <code>max_depth</code>, <code>feature_importances_</code>, <code>plot_tree</code></td><td>Clase de árboles de decisión</td></tr>
<tr><td>Métricas</td><td>Accuracy, precision, recall, F1 macro, matriz de confusión, ROC/AUC y Precision-Recall/AUC-PR en One-vs-Rest con promedio micro/macro</td><td>Clase de métricas; taller 7 (<code>auc(recall, precision)</code>)</td></tr>
<tr><td>Red neuronal</td><td>Dense, ReLU, softmax, entropía cruzada categórica con etiquetas one-hot, Adam, L2, Dropout, EarlyStopping</td><td>Clase de redes neuronales; taller 8</td></tr>
<tr><td>K-Means</td><td>Inicialización k-means++, codo con SSE (inertia), silhouette, escalado previo</td><td>Clase de K-Means; taller 7</td></tr>
<tr><td>DBSCAN</td><td>ε con la curva de k-distancia, MinPts = 2 × nº de características, silhouette sin ruido</td><td>Clase de DBSCAN; taller 7</td></tr>
</table>
<p><b>Lo que no está en el material de clase que recibimos</b> y por qué lo usamos: <b>KNN</b> y <b>SVM</b> (el enunciado los exige; la SVM va con kernel RBF y el k de KNN sale de validación cruzada de 5 folds, como pide el enunciado), <b>SMOTE</b> y el submuestreo para el balanceo, el <b>recorte por IQR</b> y la <b>PCA para graficar</b> (los nombra el enunciado) y la búsqueda de un ε menor en DBSCAN cuando el del codo deja un solo cluster (ajuste nuestro, descrito en 5.2). No usamos <code>BatchNormalization</code>, <code>ReduceLROnPlateau</code> ni métricas de validación externa como ARI, porque no se vieron en clase.</p>

<h2>7. Reproducibilidad y cómo ejecutar</h2>
<ol>
<li>Abrir <code>practica2.ipynb</code> en Google Colab y subir <code>city_day.csv</code> (o dejar que <code>kagglehub</code> lo descargue).</li>
<li>Editar solo la celda de configuración (<code>CEDULA</code>, <code>GRUPO</code>, <code>EQUIPO</code>); el tamaño y la semilla se derivan solos.</li>
<li><i>Entorno → Ejecutar todo</i>. Al final se genera <code>salidas.zip</code> con todas las tablas y figuras de este documento.</li>
</ol>

<h2>8. Enlace al video (pitch)</h2>
<p>Video de sustentación (máx. 5 min, todos los integrantes participan): <a href="{VIDEO_URL}">{VIDEO_URL}</a></p>
<p>Conjunto de datos original: <a href="{DATASET_URL}">{DATASET_URL}</a></p>

{ANEXO}
</body></html>"""

# --- Posprocesado: numeración automática (CSS), "Fig. N" / "Table N" como en el PDF de referencia ---
html = re.sub(r"<h([23])>(?:\d+(?:\.\d+)*\.?\s+)", r"<h\1>", html)                  # quita números escritos a mano
html = html.replace("<h2>Anexo", '<h2 class="unnum">Anexo')
html = re.sub(r"<h3>([AB]\.\d)", r'<h3 class="unnum">\1', html)
html = re.sub(r"(<figcaption>)Figura [A-Z]\. ", r"\1", html)
TCAPS = [
    "Rangos del AQI (CPCB) que definen las seis clases de AQI_Bucket.",
    "Variables del conjunto de datos y uso en los modelos.",
    "Configuración de las cuatro técnicas supervisadas.",
    "Resultados de los modelos supervisados en los ocho casos (Figura 1 del enunciado).",
    "Efecto medio de cada decisión de preprocesamiento sobre el F1 macro.",
    "Recall / F1 por clase en el caso 8.",
    "Tres datos de prueba, caso 8: clase real y predicción de cada técnica.",
    "Resultados de K-Means y DBSCAN en los ocho casos (Figura 3 del enunciado).",
    "Cruce cluster vs clase real de K-Means, caso 8.",
    "Cruce cluster vs clase real de DBSCAN, caso 8.",
    "Comparación entre K-Means y DBSCAN.",
    "Tres datos de prueba, caso 8: cluster asignado por K-Means y DBSCAN.",
    "Qué técnica sale de qué clase.",
    "Tres datos de prueba en los ocho casos (supervisado).",
    "Tres datos de prueba en los ocho casos (no supervisado).",
]
_TAB = re.compile(r'<table\b[^>]*\bclass="(?:[^"]*\s)?t(?:\s[^"]*)?"[^>]*>')   # <table class="t ..."> o class="dataframe t ..."
_n = len(_TAB.findall(html))
assert _n == len(TCAPS), f"hay {_n} tablas y {len(TCAPS)} pies de tabla: actualiza TCAPS"
_it = iter(TCAPS)
html = _TAB.sub(lambda m: f'<div class="tcap">{next(_it)}</div>' + m.group(0), html)

open(f"{D}/informe.html", "w", encoding="utf-8").write(html)
chrome = next(iter(glob.glob("/opt/pw-browsers/chromium-*/chrome-linux*/chrome") + glob.glob("/usr/bin/chromium*") + glob.glob("/usr/bin/google-chrome*")), None)
if not chrome:
    sys.exit("No encontré Chromium/Chrome; abran salidas/informe.html y exporten a PDF desde el navegador.")
subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={os.path.abspath(OUT_PDF)}",
                "file://" + os.path.abspath(f"{D}/informe.html")], check=True, capture_output=True)
print("PDF generado:", OUT_PDF)
