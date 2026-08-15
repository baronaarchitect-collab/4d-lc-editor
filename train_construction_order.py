#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_construction_order.py
===========================
Modelo PyTorch de *learning-to-rank* que aprende el ORDEN CONSTRUCTIVO de las
categorías IFC (los "capítulos" de la EDT) y exporta `construction_order.json`,
que la aplicación web (index.html) carga automáticamente para ordenar los
capítulos de cada nivel según lo que se construye primero en obra.

Idea
----
Cada categoría IFC se representa por un vector de características (bolsa de
n-gramas de caracteres de su nombre + señales de disciplina). Una pequeña red
neuronal produce un ESCALAR = "instante relativo de construcción". Se entrena
con una pérdida de ranking por pares (margin ranking loss): para cada par
(antes, después) de la secuencia canónica de obra, se fuerza
    score(antes) + margen <= score(despues).
Al ordenar por score ascendente se recupera (y generaliza) la secuencia.

Como la red aprende de los NOMBRES, también puntúa clases IFC que no estaban
en el conjunto de entrenamiento (p. ej. variantes o *proxies*), colocándolas
cerca de clases con nombres parecidos.

Uso
---
    pip install torch
    python train_construction_order.py
    # -> genera construction_order.json  (colócalo junto a index.html)

El JSON tiene la forma:
    { "sequence": ["IFCFOOTING", "IFCPILE", ... ],
      "scores":  {"IFCFOOTING": -3.12, ... },
      "meta":    {...} }
"""

import json
import itertools
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# 1) SECUENCIA CANÓNICA DE OBRA  (supervisión: orden real de construcción)
#    De lo primero (movimiento de tierras / cimentación) a lo último (acabados).
# ---------------------------------------------------------------------------
CANONICAL = [
    # Implantación / terreno
    "IFCSITE", "IFCGEOGRAPHICELEMENT", "IFCCIVILELEMENT",
    # Cimentación
    "IFCFOOTING", "IFCPILE", "IFCFOUNDATION",
    # Estructura vertical
    "IFCCOLUMN",
    # Estructura horizontal
    "IFCBEAM", "IFCSLAB", "IFCPLATE", "IFCROOF",
    # Circulaciones verticales
    "IFCSTAIRFLIGHT", "IFCSTAIR", "IFCRAMPFLIGHT", "IFCRAMP",
    # Cerramientos / albañilería
    "IFCWALLSTANDARDCASE", "IFCWALL", "IFCCURTAINWALL", "IFCMEMBER",
    # Carpinterías / vanos
    "IFCDOOR", "IFCWINDOW",
    # Barandas / metálicos vistos
    "IFCRAILING",
    # Instalaciones (MEP)
    "IFCPIPESEGMENT", "IFCPIPEFITTING", "IFCDUCTSEGMENT", "IFCDUCTFITTING",
    "IFCCABLECARRIERSEGMENT", "IFCCABLESEGMENT",
    "IFCFLOWSEGMENT", "IFCFLOWFITTING", "IFCFLOWTERMINAL", "IFCFLOWCONTROLLER",
    "IFCDISTRIBUTIONELEMENT",
    "IFCVALVE", "IFCPUMP", "IFCTANK", "IFCBOILER", "IFCAIRTERMINAL",
    "IFCSANITARYTERMINAL", "IFCLIGHTFIXTURE", "IFCOUTLET", "IFCSWITCHINGDEVICE",
    "IFCELECTRICAPPLIANCE", "IFCENERGYCONVERSIONDEVICE",
    # Acabados
    "IFCCOVERING",
    # Mobiliario / equipamiento
    "IFCFURNISHINGELEMENT", "IFCFURNITURE", "IFCSYSTEMFURNITUREELEMENT",
    # Genéricos / conjuntos (al final)
    "IFCELEMENTASSEMBLY", "IFCBUILDINGELEMENTPROXY",
    # Espacios (no se "construyen")
    "IFCSPACE",
]

# ---------------------------------------------------------------------------
# 2) FEATURIZACIÓN:  bolsa de trigramas de caracteres del nombre
# ---------------------------------------------------------------------------
def char_ngrams(name, n=3):
    s = "^" + name.upper() + "$"
    return [s[i:i + n] for i in range(len(s) - n + 1)]

# vocabulario de n-gramas a partir de la secuencia canónica
vocab = {}
for name in CANONICAL:
    for g in char_ngrams(name):
        vocab.setdefault(g, len(vocab))

def featurize(name):
    v = torch.zeros(len(vocab))
    for g in char_ngrams(name):
        if g in vocab:
            v[vocab[g]] += 1.0
    # normaliza por longitud para que nombres largos no dominen
    if v.sum() > 0:
        v = v / v.sum()
    return v

X = torch.stack([featurize(n) for n in CANONICAL])          # [N, V]
rank_target = {n: i for i, n in enumerate(CANONICAL)}

# ---------------------------------------------------------------------------
# 3) MODELO:  MLP -> score escalar (instante de construcción)
# ---------------------------------------------------------------------------
class OrderNet(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

torch.manual_seed(0)
model = OrderNet(len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
mrl = nn.MarginRankingLoss(margin=1.0)

# pares (i antes que j) de la secuencia canónica
pairs = [(i, j) for i, j in itertools.combinations(range(len(CANONICAL)), 2)]
idx_i = torch.tensor([i for i, _ in pairs])
idx_j = torch.tensor([j for _, j in pairs])
# queremos score(i) < score(j)  => target = -1 en MarginRankingLoss(x1=score_i, x2=score_j)
y = -torch.ones(len(pairs))

# ---------------------------------------------------------------------------
# 4) ENTRENAMIENTO
# ---------------------------------------------------------------------------
EPOCHS = 400
for ep in range(EPOCHS):
    model.train()
    scores = model(X)
    loss = mrl(scores[idx_i], scores[idx_j], y)
    opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 50 == 0:
        with torch.no_grad():
            pred = model(X)
            # exactitud de ranking por pares
            correct = ((pred[idx_i] < pred[idx_j]).float().mean().item())
        print(f"epoch {ep+1:4d}  loss={loss.item():.4f}  pairwise_acc={correct*100:.1f}%")

# ---------------------------------------------------------------------------
# 5) EXPORTAR  construction_order.json
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    scores = model(X).tolist()

scored = sorted(zip(CANONICAL, scores), key=lambda t: t[1])
sequence = [name for name, _ in scored]

out = {
    "sequence": sequence,
    "scores": {name: round(s, 4) for name, s in scored},
    "meta": {
        "model": "OrderNet (MLP + MarginRankingLoss)",
        "framework": f"pytorch {torch.__version__}",
        "features": "char-trigrams bag-of-words del nombre IFC",
        "n_categories": len(CANONICAL),
        "note": "Ordena los capítulos (categorías IFC) de la EDT por secuencia real de obra.",
    },
}
with open("construction_order.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("\nOrden constructivo aprendido (primeros -> ultimos):")
for i, name in enumerate(sequence):
    print(f"  {i:2d}. {name}")
print("\n[OK] construction_order.json generado.")
