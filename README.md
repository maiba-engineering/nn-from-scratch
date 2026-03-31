# Neural Networks from Scratch

My implementation notes and code from Andrej Karpathy's ["Neural Networks: Zero to Hero"](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) series. Everything is written from scratch in Python/PyTorch to understand how neural networks and language models actually work under the hood.

I'm an engineering student at CentraleSupélec. I started this as a self-teaching project before my official ML coursework because I wanted to understand LLMs deeply — not just use APIs.

## What's in here

| Folder | Video | What I built | Key concept |
|---|---|---|---|
| `01_micrograd/` | Intro to backpropagation | Autograd engine from scratch | How gradients flow backwards |
| `02_bigram/` | Intro to language modeling | Bigram character-level model | How language models predict the next token |
| `03_mlp/` | MLP + activations + BatchNorm | Multi-layer perceptron for text | Training, overfitting, learning rate tuning |
| `04_gpt/` | Let's build GPT | **Full Transformer from scratch** | Attention, multi-head, positional encoding |

## The journey

I watched the videos in order over a few weeks. For each one, I first watched the full video taking notes, then re-implemented everything myself without looking at the code. When something didn't work (which happened a lot), I went back to compare with Karpathy's version to find my mistakes.

The most important thing I learned: watching someone code a neural net and actually coding one yourself are two very different things. The bugs you encounter when you do it yourself — wrong tensor shapes, broken gradients, silent logical errors — are where the real learning happens.

## How to run

Each folder has its own script. Just `cd` into the folder and run:

```bash
# micrograd
cd 01_micrograd && python micrograd.py

# bigram model
cd 02_bigram && python bigram.py

# MLP
cd 03_mlp && python mlp.py

# mini-GPT (the main one)
cd 04_gpt
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
python model.py
```

## Stack

- Python 3.10+
- PyTorch (only for 02_bigram onwards — micrograd is pure Python)
- No HuggingFace, no high-level libraries

## References

- [Andrej Karpathy's YouTube](https://www.youtube.com/@andrejkarpathy)
- [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762)
- [micrograd repo](https://github.com/karpathy/micrograd)
- [makemore repo](https://github.com/karpathy/makemore)
- [nanoGPT repo](https://github.com/karpathy/nanoGPT)
