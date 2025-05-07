import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.util.display.display import Display
from typing import Dict
import networkx as nx
from itertools import combinations

class PlaylistDisplay(Display):
    """Custom display for the optimization process"""
    
    def _do(self, problem, evaluator, algorithm):
        return {
            "gen": algorithm.n_gen,
            "n_eval": evaluator.n_eval,
            "n_nds": len(algorithm.opt),
            "igd": "-",
            "temp_smooth": np.round(np.mean([s.F[0] for s in algorithm.pop]), 3),
            "energy_cont": np.round(np.mean([s.F[1] for s in algorithm.pop]), 3),
            "mood_consist": np.round(np.mean([s.F[2] for s in algorithm.pop]), 3),
            "key_compat": np.round(np.mean([s.F[3] for s in algorithm.pop]), 3),
            "genre_jump": np.round(np.mean([s.F[4] for s in algorithm.pop]), 3),
        }

class PermutationSampling(Sampling):
    """Sampling operator for permutation problems"""
    
    def _do(self, problem, n_samples, **kwargs):
        X = np.zeros((n_samples, problem.n_var), dtype=int)
        for i in range(n_samples):
            X[i, :] = np.random.permutation(problem.n_var)
        return X

class OrderCrossover(Crossover):
    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=2)

    def _do(self, problem, X, **kwargs):
        _, n_matings, n_var = X.shape
        Y = np.full((self.n_offsprings, n_matings, n_var), -1, dtype=int)

        for k in range(n_matings):
            p1, p2 = X[0, k, :], X[1, k, :]

            for off in range(self.n_offsprings):
                # decide parents a, b for this offspring
                if off == 0:
                    a, b = p1, p2
                else:
                    a, b = p2, p1

                # pick two cut points
                start, end = np.sort(np.random.choice(n_var, 2, replace=False))

                # 1) copy the slice from a into the offspring
                Y[off, k, start:end+1] = a[start:end+1]

                # 2) fill the rest from b in order
                remaining = [gene for gene in b if gene not in a[start:end+1]]
                idx = 0

                # tail
                for pos in range(end+1, n_var):
                    Y[off, k, pos] = remaining[idx]
                    idx += 1

                # head
                for pos in range(0, start):
                    Y[off, k, pos] = remaining[idx]
                    idx += 1

        return Y

class SwapMutation(Mutation):
    """Swap Mutation for permutation problems"""
    
    def __init__(self, prob=1.0):
        super().__init__()
        self.prob = prob
    
    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        
        for i in range(len(X)):
            if np.random.random() < self.prob:
                idx1, idx2 = np.random.choice(range(problem.n_var), 2, replace=False)
                Y[i, idx1], Y[i, idx2] = Y[i, idx2], Y[i, idx1]
        
        return Y

class PlaylistOptimizationProblem(Problem):
    """NSGA-II Problem formulation for playlist optimization"""
    
    def __init__(self, songs, playlist_size):
        self.songs = songs
        
        super().__init__(n_var=playlist_size,
                         n_obj=5,
                         xl=0, xu=len(self.songs)-1,
                         vtype=int,
                         vectorized=True)
        
        self.key_distance_matrix = self._create_key_distance_matrix()
    
    def _evaluate(self, x, out, *args, **kwargs):
        pop_size = x.shape[0]
        F = np.zeros((pop_size, 5))
        for i in range(pop_size):
            # Extract that one permutation:
            perm = x[i].astype(int)         # now an array of ints
            playlist = [self.songs[idx] for idx in perm]

            F[i, 0] = self._calculate_tempo_smoothness(playlist)
            F[i, 1] = self._calculate_energy_continuity(playlist)
            F[i, 2] = self._calculate_mood_consistency(playlist)
            F[i, 3] = self._calculate_key_compatibility(playlist)
            F[i, 4] = self._calculate_genre_jump(playlist)

        out["F"] = F
    
    def _calculate_tempo_smoothness(self, playlist) -> float:
        """
        Calculate how smoothly tempo transitions from one song to the next.
        Lower values indicate smoother transitions.
        """
        tempo_changes = []
        for i in range(len(playlist) - 1):
            tempo_changes.append(abs(playlist[i].tempo - playlist[i+1].tempo))
        return np.mean(tempo_changes) if tempo_changes else 0.0
    
    def _calculate_energy_continuity(self, playlist) -> float:
        """
        Calculate how smoothly energy levels transition.
        Lower values indicate smoother transitions.
        """
        energy_changes = []
        for i in range(len(playlist) - 1):
            energy_changes.append(abs(playlist[i].energy - playlist[i+1].energy))
        return np.mean(energy_changes) if energy_changes else 0.0
    
    def _calculate_mood_consistency(self, playlist) -> float:
        """
        Calculate consistency of mood (valence) throughout the playlist.
        Lower values indicate more consistent mood.
        """
        valence_changes = []
        for i in range(len(playlist) - 1):
            valence_changes.append(abs(playlist[i].valence - playlist[i+1].valence))
        return np.mean(valence_changes) if valence_changes else 0.0
    
    def _create_key_distance_matrix(self) -> np.ndarray:
        """
        Create a matrix of distances between musical keys based on the circle of fifths.
        The circle of fifths represents key relationships in music theory.
        """
        # Create a circle of fifths graph
        G = nx.Graph()
        keys = list(range(12))  # 0 to 11 representing C, C#, D, etc.
        
        # Add nodes
        for key in keys:
            G.add_node(key)
        
        # Add edges based on the circle of fifths
        for key in keys:
            # Add edge to the perfect fifth above (7 semitones)
            fifth_above = (key + 7) % 12
            G.add_edge(key, fifth_above, weight=1)
        
        # Calculate shortest path distances between all pairs of keys
        distances = np.zeros((12, 12))
        for k1, k2 in combinations(keys, 2):
            dist = nx.shortest_path_length(G, k1, k2, weight='weight')
            distances[k1, k2] = distances[k2, k1] = dist
        
        # Normalize distances to [0, 1]
        max_dist = np.max(distances)
        if max_dist > 0:
            distances /= max_dist
        
        return distances
    
    def _calculate_key_compatibility(self, playlist) -> float:
        """
        Calculate compatibility between adjacent songs based on their musical keys.
        Uses the circle of fifths to determine key relationships.
        Lower values indicate better key compatibility.
        """
        key_changes = []
        for i in range(len(playlist) - 1):
            key1 = int(playlist[i].key)
            key2 = int(playlist[i+1].key)
            
            # Handle invalid keys (-1)
            if key1 == -1 or key2 == -1:
                # Use average distance when key is unknown
                key_changes.append(0.5)
            else:
                # Use pre-calculated distance
                key_changes.append(self.key_distance_matrix[key1, key2])
        
        return np.mean(key_changes) if key_changes else 0.0
    
    def _calculate_genre_jump(self, playlist) -> float:
        jumps = []
        for i in range(len(playlist)-1):
            # Parse genres (assuming track_genre is a string of genres separated by semicolons)
            g1 = set(playlist[i].track_genre.split(';')) if playlist[i].track_genre else set()
            g2 = set(playlist[i+1].track_genre.split(';')) if playlist[i+1].track_genre else set()
            
            if not g1 and not g2:
                jumps.append(0.5)
            else:
                # Calculate Jaccard similarity and convert to distance
                jumps.append(1 - len(g1 & g2) / len(g1 | g2) if (g1 | g2) else 0.5)
        return float(np.mean(jumps)) if jumps else 0.0

def weighted_sum_approach(pareto_front, weights: Dict[str, float]) -> int:
    """
    Select a solution from the Pareto front using a weighted sum approach.
    
    Args:
        pareto_front: List of non-dominated solutions
        weights: Dictionary mapping objective names to weights
    
    Returns:
        Index of the selected solution
    """
    # Ensure weights sum to 1.0
    weight_sum = sum(weights.values())
    normalized_weights = {k: v/weight_sum for k, v in weights.items()} if weight_sum > 0 else weights
    
    objective_names = ["tempo_smoothness", "energy_continuity", "mood_consistency", "key_compatibility", "genre_jump_smoothness"]
    
    min_score = float('inf')
    best_idx = 0
    
    # Calculate weighted sum for each solution
    for idx, solution in enumerate(pareto_front):
        weighted_sum = 0.0
        for i, obj_name in enumerate(objective_names):
            weighted_sum += solution.F[i] * normalized_weights[obj_name]
        
        if weighted_sum < min_score:
            min_score = weighted_sum
            best_idx = idx
    
    return best_idx

def run_optimization(songs, weights, pop_size=100, n_gen=50):
    """
    Run the NSGA-II optimization algorithm.
    
    Args:
        songs: List of song objects
        weights: Dictionary of weights for the different objectives
        pop_size: Population size
        n_gen: Number of generations
    
    Returns:
        Optimized playlist
    """
    playlist_size = len(songs)
    problem = PlaylistOptimizationProblem(songs, playlist_size)
    
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=PermutationSampling(),
        crossover=OrderCrossover(),
        mutation=SwapMutation(prob=0.3),
        eliminate_duplicates=True,
        display=PlaylistDisplay()
    )
    
    res = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=True
    )
    
    # Choose the best solution based on the provided weights
    weights_dict = {
        "tempo_smoothness": weights["tempo"],
        "energy_continuity": weights["energy"],
        "mood_consistency": weights["mood"],
        "key_compatibility": weights["key"],
        "genre_jump_smoothness": weights["genre"]
    }
    
    selected_idx = weighted_sum_approach(res.opt, weights_dict)
    selected_solution = res.opt[selected_idx]
    
    # Return the ordered song list
    return [songs[idx] for idx in selected_solution.X]