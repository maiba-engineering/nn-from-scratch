"""
micrograd.py — Moteur d'autograd from scratch (vidéo 1 de Karpathy)

chaque opération mathématique (addition, multiplication, etc.)
sait calculer son gradient local. En enchaînant ces gradients du bout
à l'origine (la "chain rule"), on obtient le gradient de la loss par
rapport à chaque paramètre. C'est ce gradient qui dit dans quelle
direction ajuster les poids pour que le modèle se trompe moins.

Ce fichier implémente une classe Value qui :
1. Stocke un nombre
2. Garde la trace de toutes les opérations qui l'ont produit (le "graph")
3. Sait calculer les gradients en remontant ce graph (backward)

C'est exactement ce que fait PyTorch avec torch.Tensor et .backward(),
mais ici tout est écrit à la main pour comprendre.

Référence : https://www.youtube.com/watch?v=VMj-3S1tku0
"""

import math


class Value:
    """Un nombre qui sait calculer ses gradients."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0  # gradient, initialisé à 0

        # pour la backpropagation : qui m'a créé et comment ?
        self._backward = lambda: None  # fonction de calcul du gradient local
        self._prev = set(_children)    # les Values dont je dépends
        self._op = _op                 # l'opération qui m'a créé (pour le debug)

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        # gradient de l'addition : d(a+b)/da = 1, d(a+b)/db = 1
        # le += est important : si une Value est utilisée plusieurs fois,
        # les gradients s'accumulent (c'est la multivariate chain rule)
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        # gradient de la multiplication : d(a*b)/da = b, d(a*b)/db = a
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        out = Value(self.data ** other, (self,), f"**{other}")

        # gradient de x^n : n * x^(n-1)
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        """Fonction d'activation tanh."""
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        # gradient de tanh : 1 - tanh²
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        """Fonction d'activation ReLU."""
        out = Value(max(0, self.data), (self,), "relu")

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Backpropagation : calcule les gradients de bout en bout.

        On parcourt le graph en ordre topologique inversé
        (de la sortie vers les entrées) et on appelle _backward()
        sur chaque nœud.
        """
        # tri topologique
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # backprop : le gradient de la sortie par rapport à elle-même = 1
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # opérations utilitaires pour que Python gère a + 2, 2 + a, etc.
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other**-1


# ============================================================
# NEURONE, COUCHE, et MLP
# ============================================================

class Neuron:
    """Un seul neurone : somme pondérée + activation."""

    def __init__(self, n_inputs):
        # poids aléatoires entre -1 et 1
        import random
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        # somme pondérée : w1*x1 + w2*x2 + ... + b
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    """Une couche de neurones."""

    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    """Multi-Layer Perceptron."""

    def __init__(self, n_inputs, layer_sizes):
        sizes = [n_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(layer_sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("=== Test backpropagation ===\n")

    # test simple : vérifier que les gradients sont corrects
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + b**2
    c.backward()

    print(f"a = {a}")
    print(f"b = {b}")
    print(f"c = a*b + b² = {c}")
    print(f"dc/da = {a.grad} (attendu: 3.0)")   # d(a*b + b²)/da = b = 3
    print(f"dc/db = {b.grad} (attendu: 8.0)")   # d(a*b + b²)/db = a + 2b = 2+6 = 8

    print("\n=== Mini réseau de neurones ===\n")

    # un petit MLP qui apprend le XOR
    # (XOR est le hello world des réseaux de neurones)
    model = MLP(2, [4, 4, 1])
    print(f"Paramètres : {len(model.parameters())}")

    # données XOR
    xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    ys = [-1, 1, 1, -1]  # -1 au lieu de 0 car on utilise tanh

    # entraînement
    for epoch in range(200):
        # forward
        preds = [model(x) for x in xs]
        loss = sum((p - y)**2 for p, y in zip(preds, ys))

        # backward
        for p in model.parameters():
            p.grad = 0.0
        loss.backward()

        # update (descente de gradient)
        lr = 0.05
        for p in model.parameters():
            p.data -= lr * p.grad

        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.data:.4f}")

    print(f"\nPrédictions finales :")
    for x, y in zip(xs, ys):
        pred = model(x)
        print(f"  {x} → {pred.data:+.3f} (attendu: {y:+d})")
