"""
mlp.py — MLP character-level language model (vidéos 3-5 de Karpathy)

Évolution du bigram : au lieu de regarder UN seul caractère pour prédire
le suivant, on regarde les N derniers caractères (le "contexte").


Concepts introduits dans les vidéos 3-5 :
- Embeddings : transformer un index en vecteur de nombres
- Train/val split : séparer les données pour détecter l'overfitting
- Learning rate : comment le choisir (trop grand → diverge, trop petit → trop lent)
- Batch Normalization : stabiliser l'entraînement
- Backprop manuelle : comprendre comment les gradients circulent

Référence :
  - Vidéo 3 : https://www.youtube.com/watch?v=TCH_1BHY58I
  - Vidéo 4 : https://www.youtube.com/watch?v=P6sfmUTpUmc
  - Vidéo 5 : https://www.youtube.com/watch?v=t3YJ5hKiMQ0
"""

import torch
import torch.nn.functional as F


# ============================================================
# CONFIG
# ============================================================

CONTEXT_SIZE = 8     # combien de caractères le modèle regarde en arrière
N_EMBD = 24          # dimension des embeddings par caractère
N_HIDDEN = 128       # taille de la couche cachée
BATCH_SIZE = 64
MAX_STEPS = 5000
LR = 0.01


# ============================================================
# DATA
# ============================================================

def load_and_prepare(filepath):
    """Charge le texte et crée les paires (contexte → prochain caractère)."""
    try:
        with open(filepath, "r") as f:
            text = f.read()
    except FileNotFoundError:
        text = "to be or not to be that is the question " * 500
        print("(dataset pas trouvé, utilisation d'un texte de secours)")

    chars = sorted(list(set(text)))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)
    print(f"Vocabulaire : {vocab_size} caractères")

    # construire les exemples d'entraînement
    # pour chaque position dans le texte, le contexte = les N caractères précédents
    # et le target = le caractère suivant
    data = [stoi[c] for c in text]
    X, Y = [], []
    for i in range(CONTEXT_SIZE, len(data)):
        context = data[i - CONTEXT_SIZE:i]
        target = data[i]
        X.append(context)
        Y.append(target)

    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)

    # split train / val (90% / 10%)
    n = int(0.9 * len(X))
    return X[:n], Y[:n], X[n:], Y[n:], vocab_size, itos


# ============================================================
# MODÈLE MLP
# ============================================================

class CharMLP:
    """MLP character-level language model."""

    def __init__(self, vocab_size):
        # embedding : chaque caractère → vecteur de dimension N_EMBD
        self.C = torch.randn((vocab_size, N_EMBD))

        # couche 1 : prend les N embeddings concaténés → couche cachée
        # input size = CONTEXT_SIZE * N_EMBD (tous les embeddings mis bout à bout)
        self.W1 = torch.randn((CONTEXT_SIZE * N_EMBD, N_HIDDEN)) * 0.01
        self.b1 = torch.zeros(N_HIDDEN)

        # batch norm (stabilise l'entraînement, vidéo 4)
        self.bn_gain = torch.ones(N_HIDDEN)
        self.bn_bias = torch.zeros(N_HIDDEN)
        self.bn_mean_running = torch.zeros(N_HIDDEN)
        self.bn_std_running = torch.ones(N_HIDDEN)

        # couche 2 : couche cachée → probabilités sur le vocabulaire
        self.W2 = torch.randn((N_HIDDEN, vocab_size)) * 0.01
        self.b2 = torch.zeros(vocab_size)

        # activer les gradients sur tous les paramètres
        self.parameters = [self.C, self.W1, self.b1, self.bn_gain, self.bn_bias, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True

        n_params = sum(p.nelement() for p in self.parameters)
        print(f"Modèle : {n_params:,} paramètres")

    def forward(self, X, training=True):
        # lookup des embeddings : chaque index → vecteur
        emb = self.C[X]  # (batch, context_size, n_embd)

        # concaténer les embeddings en un seul vecteur
        h = emb.view(emb.shape[0], -1)  # (batch, context_size * n_embd)

        # couche 1 + batch norm + tanh
        h = h @ self.W1 + self.b1

        # batch normalization : normaliser les activations
        # pour que chaque neurone ait une moyenne ~0 et un écart-type ~1
        # ça stabilise l'entraînement et permet d'utiliser des learning rates plus grands
        if training:
            bn_mean = h.mean(dim=0)
            bn_std = h.std(dim=0)
            # mettre à jour les stats running pour l'inférence
            with torch.no_grad():
                self.bn_mean_running = 0.999 * self.bn_mean_running + 0.001 * bn_mean
                self.bn_std_running = 0.999 * self.bn_std_running + 0.001 * bn_std
        else:
            bn_mean = self.bn_mean_running
            bn_std = self.bn_std_running

        h = self.bn_gain * (h - bn_mean) / (bn_std + 1e-5) + self.bn_bias
        h = torch.tanh(h)

        # couche 2 → logits
        logits = h @ self.W2 + self.b2
        return logits


# ============================================================
# ENTRAÎNEMENT & GÉNÉRATION
# ============================================================

def main():
    X_train, Y_train, X_val, Y_val, vocab_size, itos = load_and_prepare("../04_gpt/input.txt")
    print(f"Train : {len(X_train):,} exemples | Val : {len(X_val):,}")

    model = CharMLP(vocab_size)

    print(f"\nEntraînement ({MAX_STEPS} steps)")
    print("-" * 40)

    for step in range(MAX_STEPS):
        # mini-batch aléatoire
        ix = torch.randint(0, len(X_train), (BATCH_SIZE,))
        xb, yb = X_train[ix], Y_train[ix]

        # forward
        logits = model.forward(xb, training=True)
        loss = F.cross_entropy(logits, yb)

        # backward
        for p in model.parameters:
            p.grad = None
        loss.backward()

        # update (avec decay du learning rate)
        lr = LR if step < MAX_STEPS * 0.7 else LR * 0.1
        for p in model.parameters:
            p.data -= lr * p.grad

        if step % (MAX_STEPS // 10) == 0:
            # évaluer sur le val set
            with torch.no_grad():
                val_logits = model.forward(X_val[:1000], training=False)
                val_loss = F.cross_entropy(val_logits, Y_val[:1000])
            print(f"  Step {step:5d} | train: {loss.item():.4f} | val: {val_loss.item():.4f}")

    # générer du texte
    print(f"\nGénération :")
    print("-" * 40)
    context = [0] * CONTEXT_SIZE
    result = []
    for _ in range(300):
        x = torch.tensor([context], dtype=torch.long)
        logits = model.forward(x, training=False)
        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        context = context[1:] + [idx]
        result.append(itos[idx])
    print("".join(result))


if __name__ == "__main__":
    main()
