"""Partie 2 : PageRank personnalisé sur le graphe de navigation Wikispeedia.

Utilisation :
    python pagerank_personnalise.py [chemin/vers/paths_finished.tsv]
"""

import os
import sys
import time

import numpy as np

from pagerank_standard import create_node_index, load_paths, matrice_transition


def pagerank_personnalise(matrix, beta, personal_nodes, nodes, max_iterations=100, epsilon=1e-6):
    """PageRank personnalisé : la téléportation favorise un sous-ensemble de pages."""
    n = matrix.shape[0]

    # Initialisation
    pagerank = np.ones(n) / n
    personal_vector = np.zeros(n)

    # 1. Construction du vecteur de personnalisation
    found_nodes = 0
    for node in personal_nodes:
        if node in nodes:
            personal_vector[nodes[node]] = 1
            found_nodes += 1

    if found_nodes == 0:
        personal_vector = np.ones(n) / n
    else:
        personal_vector /= np.sum(personal_vector)

    iteration = 0
    debut = time.time()
    diff = 1

    # La matrice de transition est déjà orientée « liens sortants en colonne »
    # (voir matrice_transition) : on l'utilise telle quelle. Une transposition
    # supplémentaire inverserait le sens des liens.
    while iteration < max_iterations and diff >= epsilon:
        previous_pagerank = pagerank.copy()

        # Formule du PageRank personnalisé : P = (1 - beta) * V + beta * M * P
        pagerank = (1 - beta) * personal_vector + beta * (matrix @ previous_pagerank)

        # Renormalisation pour gérer les puits (dead ends)
        s = np.sum(pagerank)
        if s > 0:
            pagerank /= s

        diff = np.linalg.norm(pagerank - previous_pagerank, ord=1)
        iteration += 1

    temps_exec = time.time() - debut
    return pagerank, iteration, temps_exec


def afficher_top_k(pagerank_vector, nodes_dict, k=20, titre="Résultats"):
    """Affiche les k pages les mieux classées."""
    idx_to_node = {v: key for key, v in nodes_dict.items()}

    # Tri des indices par score (ordre croissant), on prend les k derniers et on inverse
    top_indices = np.argsort(pagerank_vector)[-k:][::-1]

    print(f"\n=== Top {k} : {titre} ===")
    print(f"{'Rang':<6} {'Page':<35} {'Score':<15}")
    print("-" * 60)

    for i, idx in enumerate(top_indices):
        nom_page = idx_to_node.get(idx, "Inconnu")
        score = pagerank_vector[idx]
        print(f"{i + 1:<6} {nom_page:<35} {score:.6f}")


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "paths_finished.tsv"
    if not os.path.exists(file_path):
        print(f"Erreur : le fichier {file_path} est introuvable.")
        sys.exit(1)

    print("Chargement et construction de la matrice...")
    paths = load_paths(file_path)
    nodes = create_node_index(paths)
    P = matrice_transition(paths, nodes)

    beta = 0.85

    # 1. Thème « Computer »
    target_nodes = ["Computer"]
    print(f"\nCalcul du PageRank personnalisé pour {target_nodes}...")
    pr_computer, it1, t1 = pagerank_personnalise(P, beta, target_nodes, nodes)
    afficher_top_k(pr_computer, nodes, k=20, titre="Thème 'Computer'")

    # 2. Thème « Savane »
    target_nodes_savane = ["Lion", "Zebra"]
    print(f"\nCalcul du PageRank personnalisé pour {target_nodes_savane}...")
    pr_savane, it2, t2 = pagerank_personnalise(P, beta, target_nodes_savane, nodes)
    afficher_top_k(pr_savane, nodes, k=20, titre="Thème 'Savane' (Lion, Zebra)")

    # 3. Résumé des performances
    print("\n=== Résumé des performances ===")
    print(f"{'Scénario':<25} {'Itérations':<15} {'Temps (s)':<15}")
    print("-" * 55)
    print(f"{'Computer':<25} {it1:<15} {t1:<15.4f}")
    print(f"{'Savane':<25} {it2:<15} {t2:<15.4f}")
