"""
model.py — Mini-GPT from scratch avec PyTorch
tutoriel "Let's build GPT" d'Andrej Karpathy.
https://www.youtube.com/watch?v=kCc8FmEb1nY

use:
    python model.py                    # entraîne et génère du texte
    python model.py --n_heads 8        # expérimenter avec 8 têtes d'attention
    python model.py --n_embd 128       # expérimenter avec des embeddings de dim 128
    python model.py --n_layer 6        # expérimenter avec 6 blocs Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
from pathlib import Path


class CharTokenizer:

    def __init__(self, text):
        # trouver tous les caractères uniques dans le texte
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # créer les mappings caractère ↔ index
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        self.idx_to_char = {i: c for i, c in enumerate(self.chars)}

        print(f"Vocabulaire : {self.vocab_size} caractères uniques")
        print(f"Caractères : {''.join(self.chars[:50])}...")

    def encode(self, text):
        """Texte → liste d'indices. Ex: "abc" → [0, 1, 2]"""
        return [self.char_to_idx[c] for c in text]

    def decode(self, indices):
        """Liste d'indices → texte. Ex: [0, 1, 2] → "abc" """
        return "".join([self.idx_to_char[i] for i in indices])



class SelfAttentionHead(nn.Module):

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        # projections linéaires pour Q, K, V
        # pas de bias, comme dans le papier original
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # masque causal : matrice triangulaire inférieure
        # les 1 en bas = "autorisé à regarder", les 0 en haut = "interdit"
        #
        # Exemple pour une séquence de 4 tokens :
        # [[1, 0, 0, 0],    token 0 ne voit que lui-même
        #  [1, 1, 0, 0],    token 1 voit tokens 0 et 1
        #  [1, 1, 1, 0],    token 2 voit tokens 0, 1, 2
        #  [1, 1, 1, 1]]    token 3 voit tout
        #
        # sans torch.tril, le modèle voyait tout et apprenait à copier au lieu de prédire.
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape  # batch, sequence length, embedding dim

        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # calcul des scores d'attention : Q · K^T / √d
        # le scaling par √head_size empêche les scores de devenir trop grands
        # (ce qui rendrait le softmax trop "piqué" et les gradients instables)
        scores = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)  # (B, T, T)

        # appliquer le masque causal : mettre -inf là où mask == 0
        # softmax(-inf) = 0, donc ces positions sont ignorées
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        # softmax pour avoir des probabilités (somme = 1 par ligne)
        weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # multiplier par V pour obtenir la sortie
        out = weights @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        assert n_embd % n_heads == 0, f"n_embd ({n_embd}) doit être divisible par n_heads ({n_heads})"

        head_size = n_embd // n_heads

        # créer n_heads têtes indépendantes
        self.heads = nn.ModuleList([
            SelfAttentionHead(n_embd, head_size, block_size)
            for _ in range(n_heads)
        ])

        # projection de sortie après concaténation
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        # chaque tête traite l'input indépendamment
        # puis on concatène les résultats
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out




class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)




class TransformerBlock(nn.Module):

    def __init__(self, n_embd, n_heads, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attention = MultiHeadAttention(n_embd, n_heads, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # residual connection : x + attention(norm(x))
        x = x + self.attention(self.ln1(x))
        # residual connection : x + ffwd(norm(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class MiniGPT(nn.Module):

    def __init__(self, vocab_size, n_embd, n_heads, n_layer, block_size):
        super().__init__()

        self.block_size = block_size

        # embedding des tokens : chaque token → vecteur de dimension n_embd
        self.token_embedding = nn.Embedding(vocab_size, n_embd)

        # embedding des positions : chaque position dans la séquence → vecteur
        # sans ça, le modèle ne sait pas dans quel ORDRE sont les tokens
        # (l'attention est une opération sur des ensembles, pas des séquences)
        #
        # Ici on utilise des positional embeddings appris (pas sinusoïdaux).
        # C'est plus simple et ça marche bien pour des séquences courtes.
        # Pour les contextes longs, on utiliserait RoPE (Rotary Position Embedding).
        self.position_embedding = nn.Embedding(block_size, n_embd)

        # empiler N blocs Transformer
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_heads, block_size)
            for _ in range(n_layer)
        ])

        # normalisation finale
        self.ln_final = nn.LayerNorm(n_embd)

        # couche de sortie : projette vers la taille du vocabulaire
        # pour chaque position, on obtient un score (logit) par token possible
        self.output_head = nn.Linear(n_embd, vocab_size)

        # compter les paramètres
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Modèle : {n_params:,} paramètres ({n_params/1e6:.1f}M)")

    def forward(self, idx, targets=None):
        """
        idx : (B, T) tensor d'indices de tokens
        targets : (B, T) tensor des tokens attendus (pour calculer la loss)
        """
        B, T = idx.shape

        # embeddings : token + position
        tok_emb = self.token_embedding(idx)                          # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                       # (B, T, n_embd)

        # passer dans les blocs Transformer
        x = self.blocks(x)      # (B, T, n_embd)
        x = self.ln_final(x)    # (B, T, n_embd)

        # projeter vers le vocabulaire
        logits = self.output_head(x)  # (B, T, vocab_size)

        # calculer la loss si on a les targets
        loss = None
        if targets is not None:
            # reshape pour cross_entropy : (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        temperature : contrôle la "créativité"
            - temperature < 1 : plus conservateur, plus répétitif
            - temperature > 1 : plus créatif, plus chaotique
            - temperature = 1 : distribution normale
        """
        for _ in range(max_new_tokens):
            # garder seulement les block_size derniers tokens
            # (le modèle ne peut pas traiter plus que sa taille de contexte)
            idx_crop = idx[:, -self.block_size:]

            # forward pass
            logits, _ = self(idx_crop)

            # on ne s'intéresse qu'au dernier token (le prochain à prédire)
            logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # convertir les logits en probabilités
            probs = F.softmax(logits, dim=-1)

            # échantillonner le prochain token selon les probabilités
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # concaténer le nouveau token à la séquence
            idx = torch.cat([idx, next_token], dim=1)

        return idx


# ============================================================
# DATA LOADING
# ============================================================

def load_data(filepath, tokenizer, block_size, batch_size, train_ratio=0.9):
    """
    charge le texte, le tokenise, et crée des batches pour l'entraînement.
    dataset est découpé en séquences de taille block_size.
    input [t0, t1, ..., tn], donc séquence décalée d'un cran [t1, t2, ..., tn+1].
    """
    with open(filepath, "r") as f:
        text = f.read()

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Dataset : {len(data):,} tokens")

    # split train/val
    n = int(len(data) * train_ratio)
    train_data = data[:n]
    val_data = data[n:]
    print(f"Train : {len(train_data):,} | Val : {len(val_data):,}")

    return train_data, val_data


def get_batch(data, block_size, batch_size, device):
    """
    tire un lot aléatoire de séquences depuis le dataset.
    chaque séquence commence à une position aléatoire dans le texte.
    input = data[i : i+block_size]
    target = data[i+1 : i+block_size+1]  (décalé d'un token)
    """
    # positions de départ aléatoires
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # construire les séquences input et target
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)


# ============================================================
# ENTRAÎNEMENT
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, n_eval=50):
    """estime la loss sur les datasets train et val."""
    model.eval()
    losses = {}
    for name, data in [("train", train_data), ("val", val_data)]:
        batch_losses = []
        for _ in range(n_eval):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            batch_losses.append(loss.item())
        losses[name] = sum(batch_losses) / len(batch_losses)
    model.train()
    return losses


def train(model, train_data, val_data, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    print(f"\nEntraînement : {config['max_steps']} steps")
    print(f"Learning rate : {config['lr']}")
    print("-" * 50)

    start = time.time()

    for step in range(config["max_steps"]):
        # tirer un batch aléatoire
        x, y = get_batch(train_data, config["block_size"], config["batch_size"], device)

        # forward + backward
        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # évaluer de temps en temps
        if step % config["eval_every"] == 0 or step == config["max_steps"] - 1:
            losses = estimate_loss(
                model, train_data, val_data,
                config["block_size"], config["batch_size"], device
            )
            elapsed = time.time() - start
            print(f"  Step {step:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | {elapsed:.0f}s")

    print(f"\nEntraînement terminé en {time.time() - start:.0f}s")


# EXPÉRIMENTATIONS


# POINT D'ENTRÉE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini-GPT from scratch")
    parser.add_argument("--n_embd", type=int, default=64,
                        help="Dimension des embeddings (défaut: 64)")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Nombre de têtes d'attention (défaut: 4)")
    parser.add_argument("--n_layer", type=int, default=4,
                        help="Nombre de blocs Transformer (défaut: 4)")
    parser.add_argument("--block_size", type=int, default=128,
                        help="Taille du contexte en tokens (défaut: 128)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Taille du batch (défaut: 32)")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Nombre de steps d'entraînement (défaut: 3000)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (défaut: 3e-4)")
    parser.add_argument("--data", type=str, default="input.txt",
                        help="Fichier texte d'entraînement (défaut: input.txt)")
    parser.add_argument("--generate", type=int, default=500,
                        help="Nombre de caractères à générer après l'entraînement (défaut: 500)")
    args = parser.parse_args()

    # device : GPU si disponible, sinon CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device : {device}")

    # charger le texte et créer le tokenizer
    with open(args.data, "r") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)

    # config
    config = {
        "n_embd": args.n_embd,
        "n_heads": args.n_heads,
        "n_layer": args.n_layer,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "lr": args.lr,
        "eval_every": args.max_steps // 10,  # évaluer 10 fois pendant l'entraînement
    }

    print(f"\nConfig : {config}")

    # charger les données
    train_data, val_data = load_data(args.data, tokenizer, config["block_size"], config["batch_size"])

    # créer le modèle
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        n_embd=config["n_embd"],
        n_heads=config["n_heads"],
        n_layer=config["n_layer"],
        block_size=config["block_size"],
    ).to(device)

    # entraîner
    train(model, train_data, val_data, config, device)

    # générer du texte
    print(f"\n{'='*50}")
    print(f"GÉNÉRATION ({args.generate} caractères)")
    print(f"{'='*50}\n")

    # commencer avec un newline comme contexte initial
    start_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(start_tokens, max_new_tokens=args.generate)
    print(tokenizer.decode(generated[0].tolist()))

    # sauvegarder le modèle
    save_path = Path("checkpoints")
    save_path.mkdir(exist_ok=True)
    filename = f"mini_gpt_e{config['n_embd']}_h{config['n_heads']}_l{config['n_layer']}.pt"
    torch.save(model.state_dict(), save_path / filename)
    print(f"\nModèle sauvé → checkpoints/{filename}")
