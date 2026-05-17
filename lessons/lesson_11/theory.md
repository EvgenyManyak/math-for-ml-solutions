# Урок 11. Теория информации

> **Зачем это нужно:** энтропия и KL-дивергенция — это язык, на котором описывают многоклассовую классификацию, generative-модели, дистилляцию, VAE и обучение с подкреплением.

---

## 1. Энтропия

Сколько «неопределённости» в распределении `p`:

$$ H(p) = -\sum_i p_i \log p_i $$

Логарифм обычно по основанию 2 (в битах) или e (в натах).

```python
import numpy as np

def entropy(p):
    p = np.asarray(p)
    return -np.sum(p * np.log(p + 1e-12))

print(entropy([0.5, 0.5]))   # ≈ 0.693
print(entropy([0.99, 0.01])) # ≈ 0.056
print(entropy([0.25]*4))     # log(4)
```

**Интуиция:** энтропия максимальна для равномерного, минимальна для one-hot.

---

## 2. Cross-Entropy

$$ H(p, q) = -\sum_i p_i \log q_i $$

`H(p, q) ≥ H(p)`, равенство при `p = q`.

```python
def cross_entropy(p, q):
    p, q = np.asarray(p), np.asarray(q)
    return -np.sum(p * np.log(q + 1e-12))
```

**В ML `p` — это правда (one-hot), `q` — предсказание модели.** Минимизация cross-entropy = подстраивание `q` под `p`.

---

## 3. KL-дивергенция

$$ \text{KL}(p \| q) = \sum_i p_i \log \frac{p_i}{q_i} = H(p, q) - H(p) $$

Свойства:
- `KL(p || q) ≥ 0`, равенство ⟺ `p = q`.
- **Не симметрична.**
- Не метрика.

**Где встречается:**
- В VAE — `KL(q(z|x) || N(0, I))`.
- В дистилляции — KL между большой и маленькой моделью.
- В RL (PPO, TRPO) — ограничение на шаг политики.

---

## 4. Связь cross-entropy и MLE

Для one-hot правды:

$$ H(p, q) = -\log q_{y} $$

$$ \sum_i H(p_i, q_i) = -\ell(\theta) $$

**Минимизация cross-entropy = максимизация log-likelihood.**

---

## 5. Mutual Information

$$ I(X; Y) = \text{KL}(p(x, y) \| p(x)p(y)) $$

= 0 при независимости. **Применение:** feature selection, InfoNCE в contrastive learning.

---

## 6. Дифференциальная энтропия

Для `N(μ, σ²)`:

$$ h = \frac{1}{2} \log(2\pi e \sigma^2) $$

При фиксированной `Var` максимальную энтропию даёт нормальное.
