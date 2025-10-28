"""
Test pro SOMA (Self-Organizing Migrating Algorithm) - varianta AllToOne s vizualizací.
"""
import sys
import os

# Nastavení cesty pro import z kořene projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use('TkAgg')  # Explicitní backend pro zobrazení oken

from main import soma_all_to_one, ackley, sphere, rastrigin, rosenbrock, griewank, levy, get_default_bounds


def test_soma_sphere_with_visualization():
    """
    Spustí SOMA AllToOne na Sphere funkci s heatmap vizualizací.
    """
    print("=" * 60)
    print("SOMA ALLTOONE - Sphere funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    best_x, best_f = soma_all_to_one(
        objective=sphere,
        bounds=bounds,
        pop_size=10,
        path_length=3.0,
        step=0.11,
        prt=0.3,
        max_migrations=20,
        seed=42,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_soma_rastrigin_with_viz():
    """
    SOMA AllToOne na Rastrigin s vizualizací.
    """
    print("=" * 60)
    print("SOMA ALLTOONE - Rastrigin funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    
    best_x, best_f = soma_all_to_one(
        objective=rastrigin,
        bounds=bounds,
        pop_size=12,
        path_length=3.0,
        step=0.11,
        prt=0.3,
        max_migrations=25,
        seed=999,
        visualize=True,
        num_points=180,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_soma_ackley_with_viz():
    """
    SOMA AllToOne na Ackley s vizualizací.
    """
    print("=" * 60)
    print("SOMA ALLTOONE - Ackley funkce (2D)")
    print("=" * 60)
    
    bounds = [(-10.0, 10.0), (-10.0, 10.0)]
    
    best_x, best_f = soma_all_to_one(
        objective=ackley,
        bounds=bounds,
        pop_size=10,
        path_length=3.0,
        step=0.11,
        prt=0.3,
        max_migrations=20,
        seed=42,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_soma_rosenbrock_with_viz():
    """
    SOMA AllToOne na Rosenbrock s vizualizací.
    """
    print("=" * 60)
    print("SOMA ALLTOONE - Rosenbrock funkce (2D)")
    print("=" * 60)
    
    bounds = [(-5.0, 10.0), (-5.0, 10.0)]
    
    best_x, best_f = soma_all_to_one(
        objective=rosenbrock,
        bounds=bounds,
        pop_size=10,
        path_length=3.0,
        step=0.11,
        prt=0.3,
        max_migrations=30,
        seed=555,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (1, 1), f = 0")
    print()


def test_soma_griewank_with_viz():
    """
    SOMA AllToOne na Griewank s vizualizací.
    """
    print("=" * 60)
    print("SOMA ALLTOONE - Griewank funkce (2D)")
    print("=" * 60)
    
    bounds = [(-6.0, 6.0), (-6.0, 6.0)]
    
    best_x, best_f = soma_all_to_one(
        objective=griewank,
        bounds=bounds,
        pop_size=10,
        path_length=3.0,
        step=0.11,
        prt=0.3,
        max_migrations=25,
        seed=777,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (0, 0), f = 0")
    print()


def test_soma_levy_with_viz():
    """
    SOMA AllToOne na Levy s vizualizací.
    """
    print("=" * 60)
    print("SOMA ALLTOONE - Levy funkce (2D)")
    print("=" * 60)
    
    bounds = [(-3.0, 3.0), (-3.0, 3.0)]
    
    best_x, best_f = soma_all_to_one(
        objective=levy,
        bounds=bounds,
        pop_size=10,
        path_length=3.0,
        step=0.11,
        prt=0.3,
        max_migrations=25,
        seed=333,
        visualize=True,
        num_points=200,
    )
    
    print(f"\nVýsledek:")
    print(f"  Nejlepší bod: x = {best_x}")
    print(f"  Hodnota funkce: f(x) = {best_f:.8f}")
    print(f"  Globální optimum: (1, 1), f = 0")
    print()


def test_soma_sphere_quick():
    """
    Rychlý test SOMA na Sphere bez vizualizace.
    """
    print("=" * 60)
    print("SOMA ALLTOONE - Sphere funkce (rychlý test)")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    
    best_x, best_f = soma_all_to_one(
        objective=sphere,
        bounds=bounds,
        pop_size=10,
        path_length=3.0,
        step=0.11,
        prt=0.3,
        max_migrations=20,
        seed=123,
        visualize=False,
    )
    
    print(f"Výsledek: x = {best_x}, f(x) = {best_f:.8f}")
    
    # Kontrola: pro Sphere by mělo být blízko nuly
    assert best_f < 0.01, f"SOMA na Sphere selhalo: f = {best_f} > 0.01"
    print("✓ Test prošel!")
    print()


def test_soma_parameter_comparison():
    """
    Porovnání SOMA s různými parametry na Sphere funkci.
    """
    print("=" * 60)
    print("SOMA ALLTOONE - Porovnání parametrů")
    print("=" * 60)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    configs = [
        {"pop_size": 10, "path_length": 2.0, "step": 0.11, "prt": 0.3, "max_migrations": 20},
        {"pop_size": 10, "path_length": 3.0, "step": 0.11, "prt": 0.3, "max_migrations": 20},
        {"pop_size": 15, "path_length": 3.0, "step": 0.11, "prt": 0.4, "max_migrations": 20},
        {"pop_size": 10, "path_length": 3.0, "step": 0.22, "prt": 0.3, "max_migrations": 20},
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nKonfigurace {i}: pop={config['pop_size']}, PL={config['path_length']}, step={config['step']}, prt={config['prt']}, migrations={config['max_migrations']}")
        
        best_x, best_f = soma_all_to_one(
            objective=sphere,
            bounds=bounds,
            seed=42,
            visualize=False,
            **config
        )
        
        print(f"  Výsledek: x = [{best_x[0]:.6f}, {best_x[1]:.6f}], f(x) = {best_f:.8f}")
    
    print("\n✓ Porovnání dokončeno!")
    print()


if __name__ == "__main__":
    # Hlavní vizualizace na Rastrigin
    test_soma_rastrigin_with_viz()
    
    # Rychlý test bez vizualizace
    test_soma_sphere_quick()
    
    # Další vizualizace (odkomentujte pro spuštění)
    # test_soma_sphere_with_visualization()
    # test_soma_ackley_with_viz()
    # test_soma_rosenbrock_with_viz()
    # test_soma_griewank_with_viz()
    # test_soma_levy_with_viz()
    
    # Porovnání různých konfigurací
    test_soma_parameter_comparison()
    
    print("=" * 60)
    print("VŠECHNY TESTY DOKONČENY")
    print("=" * 60)
