# BIA – Cvičení 7 (Biologicky inspirované algoritmy)

Tento repozitář obsahuje řešení 7. cvičení z předmětu Biologicky inspirované algoritmy.
Cílem je implementace algoritmu Self-Organizing Migrating Algorithm verze AllToOne a vygenerování figury s heatmapou výsledné funkce.

https://michaelmachu.eu/data/pdf/bia/Exercise7.pdf

## Obsah

- `main.py` – modulové implementace funkcí a algoritmů:
  - **Testovací funkce (minimizační)**: Sphere, Schwefel, Rosenbrock, Rastrigin, Griewank, Levy, Michalewicz, Zakharov, Ackley
  - **SOMA AllToOne**: `soma_all_to_one(objective, bounds, pop_size=10, path_length=3.0, step=0.11, prt=0.3, max_migrations=20, seed=None, visualize=False, num_points=200)`
- `tests/` – testovací skript pro algoritmus SOMA

## Jak spustit testy (Windows / PowerShell)

```powershell

# SOMA AllToOne - vizualizace
python tests\soma_test.py
```

## SOMA AllToOne – stručně

- **Self-Organizing Migrating Algorithm** (SOMA) – algoritmus inspirovaný chováním migrujících skupin jedinců.
- V každé migraci:
  1. **Identifikace leadera**: Nalezne se nejlepší jedinec v populaci (leader)
  2. **Migrace jedinců**: Všichni jedinci (kromě leadera) se pohybují směrem k leaderovi
  3. **PRT vektor**: Pro každého jedince se vytvoří perturbační vektor (náhodná binární maska), který určuje které dimenze se budou měnit
  4. **Kroky po cestě**: Jedinec se pohybuje od své pozice směrem k leaderovi v malých krocích (parametr `step`)
  5. **Výběr nejlepší pozice**: Jedinec vyhodnotí fitness ve všech pozicích na své cestě a přeskočí na nejlepší
- Algoritmus končí po `max_migrations` migracích.
- **Klíčové vlastnosti**:
  - Jedinec skáče přímo na nejlepší nalezenou pozici (diskrétní přeskok, ne plynulý pohyb jako u PSO)
  - Všichni jedinci se pohybují směrem k jednomu cíli (leaderovi) → "AllToOne"
  - Silná konvergence k leaderovi, vhodné pro funkce s výrazným globálním optimem

## Parametry SOMA AllToOne

- **pop_size** (10) – Velikost populace
- **path_length** (3.0) – Jak daleko se jedinec může dostat od leadera (typicky 1.5–3.0)
  - Vyšší hodnota → větší explorace
- **step** (0.11) – Velikost kroku po cestě k leaderovi (typicky 0.11–0.33)
  - Menší step → více kroků → přesnější hledání, ale pomalejší
- **prt** (0.3) – Pravděpodobnost změny dimenze v PRT vektoru (typicky 0.1–0.4)
  - Určuje kolik dimenzí se bude měnit při migraci
- **max_migrations** (20) – Počet migrací (iterací)

## Poznámky

- Všechny implementované funkce jsou chápány jako minimalizační (menší je lepší). Pro maximalizaci lze předat `objective=lambda x: -g(x)`.
- Vizualizace je k dispozici pouze pro 2D funkce (2 parametry).
- Oba algoritmy umožňují nastavení `seed` pro reprodukovatelnost výsledků.
