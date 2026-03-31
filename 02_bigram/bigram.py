"""
bigram.py — Modèle de langage bigram (vidéo 2 de Karpathy)

Le modèle le plus simple possible : pour prédire le prochain caractère,
on regarde UNIQUEMENT le caractère actuel. Pas de contexte, pas de mémoire.

"q" → probablement suivi de "u"
"t" → probablement suivi de "h" ou "e"

C'est nul comme modèle, mais c'est le point de départ pour comprendre :
- comment on structure les données pour un modèle de langage
- comment on calcule une loss (negative log likelihood)
- comment on sample du texte depuis un modèle

Référence : https://www.youtube.com/watch?v=PaCmpygFfXo
"""

import torch
import torch.nn.functional as F


def main():
    # charger le texte
    try:
        with open("../04_gpt/input.txt", "r") as f:
            text = f.read()
    except FileNotFoundError:
        # dataset de secours
        text = "to be or not to be that is the question " * 500
        print("(dataset Shakespeare pas trouvé, utilisation d'un texte de secours)")

    # tokenizer caractère
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    print(f"Vocabulaire : {vocab_size} caractères")

    # encoder le texte
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)

    # compter les bigrams (paires de caractères consécutifs)
    # bigram_counts[i][j] = combien de fois le caractère j suit le caractère i
    bigram_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
    for i in range(len(data) - 1):
        bigram_counts[data[i], data[i+1]] += 1

    # normaliser en probabilités (chaque ligne somme à 1)
    # +1 pour le lissage (éviter les probabilités de 0)
    probs = (bigram_counts + 1) / (bigram_counts + 1).sum(dim=1, keepdim=True)

    # calculer la loss (negative log likelihood)
    # mesure à quel point le modèle est "surpris" par le texte
    log_likelihood = 0.0
    n = 0
    for i in range(len(data) - 1):
        prob = probs[data[i], data[i+1]]
        log_likelihood += torch.log(prob)
        n += 1
    nll = -log_likelihood / n
    print(f"Negative log likelihood : {nll:.4f}")

    # générer du texte
    print(f"\nTexte généré (500 caractères) :")
    print("-" * 40)
    idx = 0  # commencer par le premier caractère
    result = []
    for _ in range(500):
        p = probs[idx]
        idx = torch.multinomial(p, num_samples=1).item()
        result.append(itos[idx])
    print("".join(result))

    # version avec torch.nn (plus propre, même résultat)
    print(f"\n{'='*40}")
    print("Version avec nn.Embedding (même idée, mais apprend les poids) :")

    # au lieu de compter les bigrams, on apprend une table de lookup
    # c'est fonctionnellement identique mais ça introduit le concept
    # d'embedding et de gradient descent qu'on utilisera dans le GPT
    W = torch.randn((vocab_size, vocab_size), requires_grad=True)

    # entraînement par gradient descent
    xs = data[:-1]  # inputs
    ys = data[1:]   # targets (décalés de 1)

    for epoch in range(200):
        # forward : logits = W[input]
        logits = W[xs]
        loss = F.cross_entropy(logits, ys)

        # backward
        W.grad = None
        loss.backward()

        # update
        with torch.no_grad():
            W -= 0.1 * W.grad

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f}")

    print(f"\nLoss finale : {loss.item():.4f} (vs comptage : {nll:.4f})")


if __name__ == "__main__":
    main()
