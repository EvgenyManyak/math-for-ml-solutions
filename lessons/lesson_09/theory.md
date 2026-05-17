# Урок 09. MLE и MAP

> **Зачем это нужно:** MSE, cross-entropy, L2-регуляризация — это не магия, а MLE/MAP в гауссовой/бернуллиевой постановке. После этого урока вы будете видеть распределения сквозь любую функцию потерь.

---

## 1. Likelihood (правдоподобие)

Дано: выборка `D = {x₁, ..., x_n}` и параметрическое семейство `p(x | θ)`. Правдоподобие — вероятность увидеть именно эти данные при фиксированном `θ`:

$$ L(\theta) = p(D | \theta) = \prod_{i=1}^{n} p(x_i | \theta) $$

Произведения неудобны, поэтому работают с **log-likelihood**:

$$ \ell(\theta) = \sum_{i=1}^{n} \log p(x_i | \theta) $$

---

## 2. Maximum Likelihood Estimation (MLE)

Идея: выбираем `θ`, при котором данные наиболее вероятны.

$$ \hat\theta_{\text{MLE}} = \arg\max_\theta \sum_i \log p(x_i | \theta) $$

**Алгоритм:**
1. Записать log-likelihood.
2. Взять производную по `θ`.
3. Приравнять к нулю и решить.

---

## 3. MLE для нормального распределения

Дано: `x_i \sim N(\mu, \sigma^2)`.

$$ \ell(\mu, \sigma) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x_i - \mu)^2 $$

Дифференцируем по `μ` и `σ`:
- `∂ℓ/∂μ = 0` → `μ̂ = (1/n) Σ x_i` (выборочное среднее).
- `∂ℓ/∂σ² = 0` → `σ̂² = (1/n) Σ (x_i - μ̂)²` (выборочная дисперсия).

**Вы только что вывели формулы среднего и дисперсии из первых принципов.**

```python
import numpy as np
X = np.random.normal(loc=5, scale=2, size=1000)
mu_hat = X.mean()
sigma_hat = np.sqrt(((X - mu_hat)**2).mean())
print(mu_hat, sigma_hat)   # ≈ 5, ≈ 2
```

---

## 4. Откуда берётся MSE

Линейная регрессия: `y_i = w^T x_i + \varepsilon_i`, где `\varepsilon_i \sim N(0, \sigma^2)`. Значит `y_i | x_i \sim N(w^T x_i, \sigma^2)`.

Log-likelihood:

$$ \ell(w) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (y_i - w^T x_i)^2 $$

Максимизация `ℓ` ⇔ минимизация `\sum (y_i - w^T x_i)^2` — это **MSE**.

**Вывод:** MSE — это MLE при предположении гауссовского шума. Если шум не гауссовский — MSE может быть субоптимальной.

---

## 5. Откуда берётся cross-entropy

Бинарная классификация: `y_i | x_i \sim \text{Bernoulli}(\sigma(w^T x_i))`.

$$ \ell(w) = \sum_i \left[ y_i \log \sigma(w^T x_i) + (1 - y_i) \log(1 - \sigma(w^T x_i)) \right] $$

Максимизация `ℓ` ⇔ минимизация **binary cross-entropy**.

Для многоклассовой — softmax + categorical cross-entropy — MLE для категориального распределения.

---

## 6. Maximum A Posteriori (MAP)

MLE игнорирует априорное знание о `θ`. MAP добавляет приор:

$$ \hat\theta_{\text{MAP}} = \arg\max_\theta \left[ \sum_i \log p(x_i | \theta) + \log p(\theta) \right] $$

Если `p(θ) = N(0, τ² I)` (априорное «веса небольшие»):

$$ \log p(w) = -\frac{1}{2\tau^2} \|w\|^2 + \text{const} $$

Подставляем в MAP с MSE-likelihood и получаем:

$$ \arg\min_w \sum_i (y_i - w^T x_i)^2 + \lambda \|w\|^2 $$

**Это Ridge regression / L2-регуляризация.** Она — MAP с гауссовским априором.

Аналогично: **L1-регуляризация (Lasso) = MAP с лапласовским априором.**

---

## 7. Bias-variance из MLE

MLE-оценка `σ̂² = (1/n) Σ (x - μ̂)²` смещённая (занижает дисперсию). Несмещённая — с `1/(n-1)` вместо `1/n`.

```python
X = np.random.normal(0, 1, size=10)
np.var(X)            # ML (смещённая) — делит на n
np.var(X, ddof=1)    # несмещённая — делит на n-1
```

В ML предпочитают смещённые оценки с меньшей дисперсией (Stein paradox). Это другая большая тема.
