# Guía de Ejercicios 2 — Inteligencia Artificial — implementación en PyMC

Reimplementación completa de los ejemplos de la guía (originalmente en `pgmpy`)
usando **PyMC 6**, más la resolución del Ejercicio 1 sobre la red *Asia*.

## Instalación

```bash
pip install pymc networkx pandas matplotlib jupyterlab
```

En la guía original:

```bash
git clone https://github.com/pgmpy/pgmpy
pip install -r requirements.txt
```

## Archivos

### Notebooks (versión para clase)

| Archivo | Contenido |
|---|---|
| `guia2_pymc_estudiante.ipynb` | Notebook con **8 ejercicios** en blanco y verificación automática. Autocontenido: no importa los `.py` |
| `guia2_pymc_solucion.ipynb` | El mismo notebook resuelto y **ya ejecutado**, con todas las salidas |
| `build_notebooks.py` | Genera ambos notebooks desde una única fuente. Editar aquí, no los `.ipynb` |

> **Los notebooks trabajan por muestreo.** Toda probabilidad se estima simulando
> casos del modelo generativo y contando, y cada estimación se contrasta contra
> el valor teórico calculado a mano (regla del producto, ley de probabilidad
> total, teorema de Bayes). No hay enumeración de distribuciones conjuntas: eso
> queda en los scripts `.py`, que conservan la versión exacta como referencia.

Los ejercicios tienen celdas `check(...)` que imprimen `OK` o `REVISAR` con una
pista, de modo que el estudiante itera solo. El notebook del estudiante corre de
principio a fin sin errores aunque todo esté en blanco: los valores marcador son
válidos pero incorrectos, así que la retroalimentación llega como `REVISAR` y no
como un *traceback*.

| Ejercicio | Tema |
|---|---|
| 1 | Completar la conjunta de eventos no excluyentes |
| 2 | Regla del producto y teorema de Bayes |
| 3 | Construir la causa común |
| 4 | Construir el colisionador y observar *explaining away* |
| 5 | Predecir el resultado de `is_active_trail` antes de ejecutarlo |
| 6 | Completar las CPT de la red Asia |
| 7 | **Ejercicio 1.1 del enunciado**: las cuatro afirmaciones de independencia |
| 8 | *Explaining away* con consecuencias clínicas |

Para regenerar tras editar `build_notebooks.py`:

```bash
python build_notebooks.py          # escribe ambos .ipynb
python ejecutar.py guia2_pymc_solucion.ipynb   # ejecuta la version resuelta
```

### Scripts

| Archivo | Contenido |
|---|---|
| `pymc_pgm.py` | Utilidades: conjunta exacta, marginales, condicionales, test de independencia, d-separación |
| `s2_probabilidad.py` | Sección 2: eventos excluyentes, no excluyentes, independientes y dependientes |
| `s3_grafos.py` | Sección 3: cadena causal, causa común, efecto común, `is_active_trail`, `fit` |
| `s4_asia.py` | Sección 4: red Asia + **Ejercicio 1 resuelto** |
| `dag_figura.py` | Genera `red_asia.png` y `red_asia_pymc.png` |

```bash
python s2_probabilidad.py
python s3_grafos.py
python s4_asia.py
python dag_figura.py
```

## Diferencia conceptual con pgmpy

`pgmpy` es una biblioteca de **modelos gráficos**: representa tablas de
probabilidad conjunta y razona sobre la estructura del grafo. PyMC es una
biblioteca de **programación probabilística**: define un modelo generativo y
resuelve por muestreo. Tres cosas que pgmpy trae de fábrica no existen en PyMC,
y este código las reconstruye.

| pgmpy | Este código sobre PyMC |
|---|---|
| `JointProbabilityDistribution([...])` | `pm.Categorical` sobre los estados conjuntos |
| `d.marginal_distribution(['A'])` | `marginal(df, 'A')` |
| `d.check_independence(['A'],['B'])` | `independencia_condicional(df, 'A', 'B')` |
| `BayesianModel([('A','C'),...])` | `pm.Model` con `pm.Bernoulli` anidadas |
| `model.fit(datos)` | `pm.sample` con priors Beta sobre las CPT |
| `model.is_active_trail('A','B')` | `camino_activo(aristas, 'A','B')` (vía `networkx`) |
| `model.get_independencies()` | `independencias_graficas(aristas)` |

### La idea clave: la conjunta exacta desde el `logp`

En un modelo dirigido discreto,

```
logp(x) = Σ_i log P(x_i | pa_i)   ⟹   exp(logp(x)) = P(x_1, …, x_n)
```

Entonces basta con evaluar `model.compile_logp()` en las `2^n` configuraciones
para recuperar la tabla conjunta **exacta**, sin error de Monte Carlo. Eso hace
`enumerar_joint`. Para la red Asia son 128 configuraciones (7 variables libres,
`E` es determinística), así que todas las respuestas del ejercicio son exactas.

Los scripts también muestran la vía nativa de PyMC (`pm.observe` + `pm.sample`)
y comparan ambos resultados.

### Detalles de implementación

- **CPT como tensores indexables.** `pt.as_tensor([0.01, 0.05])[A]` da
  `P(T=1|A)`; para dos padres, `tabla[E, B]`. Es el equivalente de `TabularCPD`.
- **Nodo lógico `E = T or L`.** Se declara con `pm.Deterministic`, no como
  variable aleatoria. Si se declarara como Bernoulli con probabilidad 0/1 el
  muestreador quedaría atrapado.
- **d-separación.** Es un criterio puramente gráfico, PyMC no lo implementa;
  se usa `networkx.is_d_separator`, que es el mismo algoritmo Bayes-Ball de
  `pgmpy.is_active_trail`.

## Ejercicio 1 — respuestas

Verificadas por dos vías independientes (criterio gráfico sobre el DAG y
cálculo numérico sobre la conjunta exacta), que coinciden en las cuatro:

| Afirmación | Respuesta | Razón |
|---|---|---|
| `T ⊥ S \| D` | **Falsa** | El camino `T→E→D←B←S` se abre al condicionar en el colisionador `D` |
| `L ⊥ B \| S` | **Verdadera** | `S` bloquea la causa común; el colisionador `D` está cerrado |
| `A ⊥ S \| L` | **Verdadera** | Todos los caminos pasan por un colisionador (`E` o `D`) no observado |
| `A ⊥ S \| L, D` | **Falsa** | Observar `D` reabre `A→T→E→D←B←S` |

Comparar (3) y (4) es lo interesante del ejercicio: **agregar evidencia puede
crear dependencias**, que es exactamente la condición (1) del criterio de
separación del enunciado.

Un matiz que aparece al mirar los números de (4): la dependencia solo se observa
cuando `L=0`. Si `L=1` entonces `E = T ∨ L = 1` sea cual sea `T`, el nodo queda
saturado y el camino se corta de hecho. Es un ejemplo de distribución **no fiel**
al grafo: la d-separación es suficiente para la independencia, pero no necesaria.

## Dos observaciones sobre el enunciado

1. **`model2.get_independencies()` (sección 3).** La salida impresa en la guía
   —`(A ⊥ B | C)`, `(A ⊥ C | B)`, `(C ⊥ B | A)` y sus simétricas— corresponde a
   un grafo *sin aristas*, no al colisionador `A→C←B`. Para
   `BayesianModel([('A','C'),('B','C')])` la única independencia es `(A ⊥ B)`,
   y condicionar en `C` precisamente la destruye, que es el punto de la Figura 2.

2. **Figura 1.** El texto la describe como «cadena causal o causa común» y las
   cuatro miniaturas mezclan ambas estructuras. Vale la pena separarlas: la
   cadena `A→C→B` y la causa común `A←C→B` tienen la misma firma de
   independencias (por eso son indistinguibles con datos observacionales), pero
   son causalmente distintas. `s3_grafos.py` las trata por separado.

## Parametrización de la red Asia

| Nodo | Parámetros |
|---|---|
| `P(A=1)` | 0.01 |
| `P(S=1)` | 0.50 |
| `P(T=1\|A)` | 0.05 / 0.01 |
| `P(L=1\|S)` | 0.10 / 0.01 |
| `P(B=1\|S)` | 0.60 / 0.30 |
| `E` | `T ∨ L` (determinística) |
| `P(X=1\|E)` | 0.98 / 0.05 |
| `P(D=1\|E,B)` | 0.90 / 0.70 / 0.80 / 0.10 |
