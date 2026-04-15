import numpy as np

class simpleNN:
    def __init__(self, input, hidden, output):
        self.W1 = np.random.randn(input, hidden)
        self.W2 = np.random.randn(hidden, output)

        self.b1 = np.zeros(hidden)
        self.b2 = np.zeros(output)

    def forward(self, x):
        z1 = np.dot(x, self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return z2

    def train_step(self, state, action, reward, lr=0.01):
        """
        Standard REINFORCE policy gradient update.

        Parameters
        ----------
        state  : np.ndarray  – the game state vector at this step
        action : int         – the action index that was taken
        reward : float       – pre-computed reward for this (state, action) pair
                               (calculated in Main.py so it always reflects the
                               correct attacker/defender at that step)
        lr     : float       – learning rate
        """

        # ── forward pass ──────────────────────────────────────────────────────
        z1 = np.dot(state, self.W1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.W2) + self.b2

        temperature = 0.5
        exp = np.exp((z2 - np.max(z2)) / temperature)
        probs = exp / np.sum(exp)

        # ── policy gradient (REINFORCE) ────────────────────────────────────────
        # dZ2 = grad of cross-entropy loss w.r.t. z2
        # multiplying by -reward turns minimising loss into maximising reward
        # (negative sign = gradient *ascent* on expected reward)
        dZ2 = probs.copy()
        dZ2[action] -= 1          # cross-entropy gradient
        dZ2 *= -reward            # scale by reward; negative → ascend reward

        # ── backprop ──────────────────────────────────────────────────────────
        dW2 = np.outer(a1, dZ2)
        dB2 = dZ2

        dA1 = np.dot(self.W2, dZ2)
        dZ1 = dA1 * (1 - a1 ** 2)   # reuse a1 instead of recomputing tanh(z1)

        dW1 = np.outer(state, dZ1)
        dB1 = dZ1

        self.W2 -= lr * dW2
        self.b2 -= lr * dB2
        self.W1 -= lr * dW1
        self.b1 -= lr * dB1