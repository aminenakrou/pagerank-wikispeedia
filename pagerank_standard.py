"""Partie 1 : PageRank standard sur le graphe de navigation Wikispeedia.

Utilisation :
    python pagerank_standard.py [chemin/vers/paths_finished.tsv]

Par défaut, le fichier `paths_finished.tsv` est cherché dans le dossier courant.
"""

import os
import sys
import time
from urllib.parse import unquote  # Pour nettoyer les noms (ex : "United_States")

import matplotlib.pyplot as plt
import numpy as np


def load_paths(file_path):
    """Charge les chemins de navigation (colonne 4 du fichier TSV)."""
    if not os.path.exists(file_path):
        print(f"ERREUR : le fichier est introuvable : {file_path}")
        print("Vérifiez que le chemin est correct et que le fichier existe.")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # On cherche où commencent les données (après les lignes commençant par #)
        start_index = 0
        for i, line in enumerate(lines):
            if not line.startswith("#"):
                start_index = i
                break

        paths = []
        for line in lines[start_index:]:
            parts = line.strip().split("\t")
            # Vérification de sécurité sur la ligne
            if len(parts) > 3:
                # Nettoyage des noms (unquote) pour éviter les doublons
                path_cleaned = [unquote(node) for node in parts[3].split(";")]
                paths.append(path_cleaned)

        print(f"Succès : {len(paths)} chemins chargés.")
        return paths

    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture : {e}")
        return []


def create_node_index(paths):
    """Crée un dictionnaire {Nom_Page -> Index_Entier}."""
    nodes = {}
    index = 0
    for path in paths:
        for node in path:
            if node != "<" and node not in nodes:
                nodes[node] = index
                index += 1
    return nodes


def matrice_transition(paths, nodes):
    """Construit la matrice de transition avec gestion des 'dangling nodes'."""
    n = len(nodes)
    # float32 pour économiser de la mémoire si le graphe est gros
    matrice_adjacence = np.zeros((n, n), dtype=np.float32)

    for path in paths:
        visited = []
        for node in path:
            if node == "<":
                # Gestion du retour arrière
                if len(visited) >= 2:
                    prev_index = nodes[visited[-1]]
                    curr_index = nodes[visited[-2]]
                    matrice_adjacence[prev_index, curr_index] += 1
                    visited.pop()
                elif visited:
                    visited.pop()
            else:
                if visited:
                    prev_index = nodes[visited[-1]]
                    curr_index = nodes[node]
                    matrice_adjacence[prev_index, curr_index] += 1
                visited.append(node)

    # Transposition pour avoir les liens sortants en colonne
    matrice_trans = matrice_adjacence.T

    # Normalisation et correction des puits (dangling nodes)
    for j in range(n):
        col_sum = np.sum(matrice_trans[:, j])
        if col_sum > 0:
            matrice_trans[:, j] /= col_sum
        else:
            # Une page sans lien sortant pointe vers toutes les pages
            matrice_trans[:, j] = 1.0 / n

    return matrice_trans


def pagerank(matrix, beta, max_iterations=100, epsilon=1e-6):
    """Algorithme PageRank vectorisé (méthode de la puissance itérée)."""
    n = matrix.shape[0]
    pagerank_vec = np.ones(n) / n
    iteration = 0
    debut = time.time()

    # Le terme de téléportation est constant
    teleport_vector = (1 - beta) / n

    while iteration < max_iterations:
        prev_pagerank = pagerank_vec.copy()

        # Formule : P = (1 - beta) / n + beta * M * P
        pagerank_vec = teleport_vector + beta * (matrix @ pagerank_vec)

        # Vérification de la convergence (norme L1)
        diff = np.linalg.norm(pagerank_vec - prev_pagerank, ord=1)
        if diff < epsilon:
            break
        iteration += 1

    temps_exec = time.time() - debut
    return pagerank_vec, iteration, temps_exec


def display_top_pages(pagerank_vec, index_to_node, beta, top_n=5):
    """Affiche les pages les mieux classées."""
    sorted_indices = np.argsort(-pagerank_vec)
    top_nodes = [index_to_node[idx] for idx in sorted_indices[:top_n]]
    print(f" -> Top {top_n} (Beta={beta:.2f}) : {top_nodes}")


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "paths_finished.tsv"

    print(f"Chargement depuis : {file_path}")
    paths = load_paths(file_path)

    if paths:
        # Préparation du graphe
        print("Indexation des nœuds...")
        nodes = create_node_index(paths)
        index_to_node = {v: k for k, v in nodes.items()}
        print(f"Nombre de pages uniques : {len(nodes)}")

        print("Construction de la matrice de transition...")
        P = matrice_transition(paths, nodes)

        # Beta de 0.1 à 0.9 (on évite 1.0)
        betas = np.arange(0.1, 0.95, 0.1)
        iterations_list = []
        temps_execs_list = []

        print("Lancement des calculs PageRank...")
        for beta in betas:
            pr, iterations, exec_time = pagerank(P, beta)
            display_top_pages(pr, index_to_node, beta)
            iterations_list.append(iterations)
            temps_execs_list.append(exec_time)

        # Graphiques : itérations et temps d'exécution selon beta
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(betas, iterations_list, marker="o", linestyle="-", color="b")
        plt.xlabel("Facteur d'amortissement (Beta)")
        plt.ylabel("Nombre d'itérations avant convergence")
        plt.title("Convergence vs Beta")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(betas, temps_execs_list, marker="o", linestyle="-", color="r")
        plt.xlabel("Facteur d'amortissement (Beta)")
        plt.ylabel("Temps d'exécution (secondes)")
        plt.title("Performance vs Beta")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("performance_beta.png", dpi=120)
        plt.show()
    else:
        print("\n--- ÉCHEC ---")
        print("Le fichier 'paths_finished.tsv' est introuvable.")
        print("Téléchargez le dataset Wikispeedia (voir README) et passez son chemin en argument.")
