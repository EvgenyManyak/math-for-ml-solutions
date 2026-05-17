# Урок 05. Цепное правило и backpropagation

> **Зачем это нужно:** backprop — это просто цепное правило, применённое к вычислительному графу. Если вы поймёте его руками, всё, что делает `loss.backward()` в PyTorch, перестанет быть магией.

---

## 1. Цепное правило в одной переменной

Если `y = f(g(x))`, то:

$$ \frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x) $$

Пример: `y = sin(x²)`. Внешняя функция `sin`, внутренняя `x²`.
`dy/dx = cos(x²) · 2x`.

```python
import numpy as np

def f(x): return np.sin(x**2)
def df_analytic(x): return np.cos(x**2) * 2 * x

# Численная проверка
def df_numeric(x, h=1e-5): return (f(x+h) - f(x-h)) / (2*h)

x = 1.5
print(df_analytic(x), df_numeric(x))   # должны совпадать
```

---

## 2. Цепное правило для композиции из 3+ функций

`y = h(g(f(x)))`:

$$ \frac{dy}{dx} = h'(g(f(x))) \cdot g'(f(x)) \cdot f'(x) $$

Это **умножение градиентов на каждом слое** — ровно то, что делает backprop.

---

## 3. Вычислительный граф

Любое выражение разворачивается в граф элементарных операций. Пример: `L = (w·x + b - y)²`.

```
x ──┐
    × ──> z ──┐
w ──┘         + ──> u ──┐
              b ────────┘
                        - ──> e ──> ² ──> L
                  y ────┘
```

Чтобы найти `dL/dw`, идём от `L` назад к `w`, перемножая локальные производные.

---

## 4. Backprop руками: ручной вывод для одного нейрона

Дано: `z = wx + b`, `a = σ(z)`, `L = (a - y)²`.

Цель: найти `dL/dw`, `dL/db`.

**Forward pass:**
```python
def sigmoid(z): return 1 / (1 + np.exp(-z))

w, b, x, y = 0.5, 0.1, 2.0, 1.0
z = w * x + b
a = sigmoid(z)
L = (a - y)**2
```

**Backward pass** (применяем цепное правило шаг за шагом):
```python
dL_da = 2 * (a - y)             # производная (a-y)² по a
da_dz = a * (1 - a)              # производная сигмоиды
dz_dw = x                        # производная wx+b по w
dz_db = 1                        # производная wx+b по b

dL_dw = dL_da * da_dz * dz_dw    # цепочка
dL_db = dL_da * da_dz * dz_db

print(dL_dw, dL_db)
```

Это весь backprop. Дальше — просто больше переменных и операция `@` вместо `*`.

---

## 5. Backprop для двухслойной сети

Архитектура: `h = ReLU(W₁x + b₁)`, `o = W₂h + b₂`, `L = (o - y)²`.

```python
def relu(z): return np.maximum(0, z)
def relu_grad(z): return (z > 0).astype(float)

# Forward
z1 = W1 @ x + b1
h = relu(z1)
o = W2 @ h + b2
L = np.sum((o - y)**2)

# Backward
dL_do = 2 * (o - y)
dL_dW2 = np.outer(dL_do, h)
dL_db2 = dL_do
dL_dh = W2.T @ dL_do
dL_dz1 = dL_dh * relu_grad(z1)
dL_dW1 = np.outer(dL_dz1, x)
dL_db1 = dL_dz1
```

Это **полный backprop руками**, без `autograd`. На этом построены все фреймворки.

---

## 6. Локальные градиенты ключевых операций

Запомните таблицу — это основа backprop:

| Операция | Forward | Backward (вход) |
|----------|---------|-----------------|
| `z = x + y` | `x + y` | `dx = dz`, `dy = dz` |
| `z = x · y` | `x · y` | `dx = y · dz`, `dy = x · dz` |
| `z = max(x, y)` | `max(x, y)` | градиент идёт только в больший |
| `z = exp(x)` | `exp(x)` | `dx = exp(x) · dz` |
| `z = log(x)` | `log(x)` | `dx = (1/x) · dz` |
| `z = W @ x` | `W @ x` | `dW = dz @ x.T`, `dx = W.T @ dz` |
| `z = ReLU(x)` | `max(0, x)` | `dx = dz если x>0, иначе 0` |
