import numpy as np
import random
from itertools import combinations
from typing import List, Tuple, Optional

# --- Configuration Parameters ---
EPSILON = 1e-10
MAX_ITERATIONS = 50000
PERTURBATION_MAGNITUDE = 1e-4

# --- Data Structure ---
class Sphere:
    """Represents a sphere defined by center (x, y, z) and radius r."""
    def __init__(self, id, center, radius):
        self.id = id
        self.center = np.array(center, dtype=np.float64)
        self.radius = float(radius)

def calc_D(sphere):
    """Calculates the D term for the sphere equation: |c|^2 - r^2"""
    return np.dot(sphere.center, sphere.center) - sphere.radius**2

def is_on_surface(sphere, point, epsilon=None):
    """
    Checks if a point lies on the surface of a sphere using an epsilon tolerance.
    This is the core robustness check.
    """
    if epsilon is None:
        epsilon = EPSILON
    distance_sq = np.sum((point - sphere.center)**2)
    radius_sq = sphere.radius**2
    return abs(distance_sq - radius_sq) <= epsilon * max(distance_sq, radius_sq, 1.0)

def solve_radical_center(s1, s2, s3, epsilon=None):
    """
    Find point(s) where all three spheres intersect.
    
    Method: 
    1. Find 2D solution space from pairwise radical planes (radical axis)
    2. Find where this axis intersects both s1 and s2
    3. Return intersection point(s)
    
    For 3 spheres in general position:
    - 2 radical planes define a line (radical axis)
    - This line may intersect the spheres at points
    - We find intersection of the radical axis with the sphere surfaces
    """
    # Get the two independent radical plane equations
    c1, r1 = s1.center, s1.radius
    c2, r2 = s2.center, s2.radius
    c3, r3 = s3.center, s3.radius
    
    D1 = np.dot(c1, c1) - r1**2
    D2 = np.dot(c2, c2) - r2**2
    D3 = np.dot(c3, c3) - r3**2
    
    # Two radical planes
    # R12: 2(c2 - c1) · x = D2 - D1
    # R13: 2(c3 - c1) · x = D3 - D1
    
    A = np.array([
        2 * (c2 - c1),
        2 * (c3 - c1)
    ])
    b = np.array([D2 - D1, D3 - D1])
    
    # Compute rank and SVD
    U, S, Vt = np.linalg.svd(A)
    rank = np.sum(S > 1e-9)
    
    if rank < 2:
        # Planes are parallel or identical - degenerate
        return None
    
    # We have 2 independent planes (a line in 3D)
    # Find a particular solution and the direction vector
    x_particular = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Direction of the line is in the null space of A (2x3 → null space is 1D)
    direction = Vt[-1]  # Last singular vector (smallest singular value)
    direction = direction / (np.linalg.norm(direction) + 1e-15)
    
    # Parametrize the radical axis: x(t) = x_particular + t * direction
    # Find where it intersects s1 and s2 (should be the same points)
    
    candidates = []
    
    for sphere in [s1, s2, s3]:
        c = sphere.center
        r = sphere.radius
        
        # |x_particular + t*direction - c|^2 = r^2
        # |x_particular - c|^2 + 2t*(x_particular - c)·direction + t^2*|direction|^2 = r^2
        
        diff = x_particular - c
        a_coef = np.dot(direction, direction)
        b_coef = 2 * np.dot(diff, direction)
        c_coef = np.dot(diff, diff) - r**2
        
        discriminant = b_coef**2 - 4*a_coef*c_coef
        
        if discriminant < -EPSILON:
            # No intersection
            continue
        
        if discriminant < 0:
            discriminant = 0
        
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b_coef + sqrt_disc) / (2*a_coef + 1e-15)
        t2 = (-b_coef - sqrt_disc) / (2*a_coef + 1e-15)
        
        p1 = x_particular + t1 * direction
        p2 = x_particular + t2 * direction
        
        candidates.append(p1)
        candidates.append(p2)
    
    # Return all candidates; calculate_kmax will verify each one
    return candidates if candidates else None

def calculate_kmax(spheres, epsilon=None):
    """
    Calculates the maximum intersection order k_max for a given set of N spheres
    using the O(N^4) Radical Center method.
    """
    if epsilon is None:
        epsilon = EPSILON
    
    N = len(spheres)
    if N < 2:
        return 0

    max_order = 0

    # 1. Baseline Check: k=2 (Any intersection)
    for s1, s2 in combinations(spheres, 2):
        d_centers = np.linalg.norm(s1.center - s2.center)
        if d_centers <= s1.radius + s2.radius + epsilon:
            max_order = max(max_order, 2)

    # 2. Triplet Check: k=3 to k=N
    for s1, s2, s3 in combinations(spheres, 3):
        result = solve_radical_center(s1, s2, s3, epsilon)

        if result is not None:
            # result can be a single point or a list of candidates
            candidates = [result] if isinstance(result, np.ndarray) else result
            
            for x_star in candidates:
                if x_star is None:
                    continue
                
                # Check if x_star is on all three defining spheres
                if (is_on_surface(s1, x_star, epsilon) and
                    is_on_surface(s2, x_star, epsilon) and
                    is_on_surface(s3, x_star, epsilon)):

                    k_current = 3
                    
                    # Check remaining spheres
                    for s_check in spheres:
                        if s_check.id in [s1.id, s2.id, s3.id]:
                            continue
                        if is_on_surface(s_check, x_star, epsilon):
                            k_current += 1

                    max_order = max(max_order, k_current)

    return max_order

def generate_random_spheres(N, radius_range, center_range):
    """Generates an initial, random arrangement of N spheres."""
    spheres = []
    for i in range(N):
        center = [random.uniform(*center_range), 
                  random.uniform(*center_range), 
                  random.uniform(*center_range)]
        radius = random.uniform(*radius_range)
        spheres.append(Sphere(i, center, radius))
    return spheres

def perturb_spheres(spheres, magnitude):
    """Applies a small random perturbation to sphere centers and radii."""
    new_spheres = []
    for s in spheres:
        new_center = s.center + np.random.uniform(-magnitude, magnitude, 3)
        new_radius = max(s.radius + random.uniform(-magnitude, magnitude), magnitude)
        new_spheres.append(Sphere(s.id, new_center, new_radius))
    return new_spheres

def perturb_spheres_directional(spheres, magnitude, direction_hint=None):
    """
    Applies directed perturbation toward improving k_max alignment.
    If direction_hint is provided, biases perturbations in that direction.
    """
    new_spheres = []
    for s in spheres:
        if direction_hint is not None:
            # Bias toward the direction hint with some randomness
            bias = 0.7 * direction_hint + 0.3 * np.random.uniform(-1, 1, 3)
            bias = bias / (np.linalg.norm(bias) + 1e-10)
            new_center = s.center + magnitude * bias
        else:
            new_center = s.center + np.random.uniform(-magnitude, magnitude, 3)
        
        new_radius = max(s.radius + random.uniform(-magnitude, magnitude), magnitude)
        new_spheres.append(Sphere(s.id, new_center, new_radius))
    return new_spheres

def calculate_centroid(spheres):
    """Calculate the centroid of all sphere centers."""
    centers = np.array([s.center for s in spheres])
    return np.mean(centers, axis=0)

def randomized_search_basic(N, max_iterations, magnitude):
    """
    Basic randomized search with simple hill-climbing.
    Reference version for comparison.
    """
    print(f"--- Starting Basic Randomized Search (N={N}) ---")
    
    RADIUS_RANGE = (1.0, 5.0)
    CENTER_RANGE = (-10.0, 10.0)
    
    best_spheres = generate_random_spheres(N, RADIUS_RANGE, CENTER_RANGE)
    best_kmax = calculate_kmax(best_spheres)
    print(f"Iteration 0: k_max={best_kmax}")

    for i in range(1, max_iterations + 1):
        candidate_spheres = perturb_spheres(best_spheres, magnitude)
        candidate_kmax = calculate_kmax(candidate_spheres)

        if candidate_kmax > best_kmax:
            best_kmax = candidate_kmax
            best_spheres = candidate_spheres
            print(f"Iteration {i}: NEW MAX k_max = {best_kmax}")
        
        if i % (max_iterations // 10) == 0 and i > 0:
            print(f"Iteration {i}: Current max k_max = {best_kmax}")

    print("--- Search Complete ---")
    return best_kmax, best_spheres


def randomized_search_adaptive(N, max_iterations, magnitude):
    """
    Enhanced randomized search with adaptive strategies:
    1. Magnitude scaling based on proximity to ideal k_max
    2. Periodic random restarts with local refinement
    3. Simulated annealing to escape local minima
    4. Plateau detection with strategy switching
    """
    print(f"--- Starting Adaptive Randomized Search (N={N}) ---")
    
    RADIUS_RANGE = (1.0, 5.0)
    CENTER_RANGE = (-10.0, 10.0)
    
    best_spheres = generate_random_spheres(N, RADIUS_RANGE, CENTER_RANGE)
    best_kmax = calculate_kmax(best_spheres)
    print(f"Iteration 0: k_max={best_kmax}")

    iterations_at_plateau = 0
    plateau_threshold = max(100, max_iterations // 20)
    current_magnitude = magnitude
    best_kmax_history = [best_kmax]
    restarts = 0
    
    for i in range(1, max_iterations + 1):
        # Adaptive magnitude scaling
        if best_kmax >= N - 1:
            current_magnitude = magnitude * 0.05  # Very fine refinement
        elif best_kmax >= N - 2:
            current_magnitude = magnitude * 0.1
        elif best_kmax >= N - 3:
            current_magnitude = magnitude * 0.3
        else:
            current_magnitude = magnitude
        
        # Strategy selection
        if i % 15 == 0 and best_kmax < N - 2:
            # Periodic restart: try new random config with local refinement
            candidate_spheres = generate_random_spheres(N, RADIUS_RANGE, CENTER_RANGE)
            for _ in range(3):
                refined = perturb_spheres(candidate_spheres, current_magnitude)
                refined_kmax = calculate_kmax(refined)
                if refined_kmax >= calculate_kmax(candidate_spheres):
                    candidate_spheres = refined
            restarts += 1
        else:
            # Local perturbation
            candidate_spheres = perturb_spheres(best_spheres, current_magnitude)
        
        candidate_kmax = calculate_kmax(candidate_spheres)

        # Acceptance logic
        if candidate_kmax > best_kmax:
            best_kmax = candidate_kmax
            best_spheres = candidate_spheres
            iterations_at_plateau = 0
            print(f"Iteration {i}: NEW MAX k_max = {best_kmax}")
        else:
            iterations_at_plateau += 1
            
            # Simulated annealing escape mechanism
            if iterations_at_plateau > plateau_threshold:
                temperature = np.exp(-iterations_at_plateau / (plateau_threshold * 2))
                acceptance_prob = np.exp(-(best_kmax - candidate_kmax) / max(temperature, 0.01))
                if random.random() < acceptance_prob and candidate_kmax >= best_kmax - 1:
                    best_spheres = candidate_spheres
                    iterations_at_plateau = 0
        
        best_kmax_history.append(best_kmax)
        
        if i % (max_iterations // 10) == 0 and i > 0:
            print(f"Iteration {i}: Current max k_max = {best_kmax} (restarts: {restarts})")

    print("--- Search Complete ---")
    return best_kmax, best_spheres


def randomized_search_genetic(N, max_iterations, magnitude, population_size=10):
    """
    Population-based search combining multiple strategies:
    1. Maintain population of good configurations
    2. Local refinement of each population member
    3. Cross-breeding via sphere parameter averaging
    4. Mutation with adaptive magnitude
    """
    print(f"--- Starting Genetic Algorithm Search (N={N}, pop_size={population_size}) ---")
    
    RADIUS_RANGE = (1.0, 5.0)
    CENTER_RANGE = (-10.0, 10.0)
    
    # Initialize population
    population = []
    for _ in range(population_size):
        spheres = generate_random_spheres(N, RADIUS_RANGE, CENTER_RANGE)
        kmax = calculate_kmax(spheres)
        population.append((spheres, kmax))
    
    # Sort by k_max
    population.sort(key=lambda x: x[1], reverse=True)
    best_kmax = population[0][1]
    print(f"Iteration 0: Initial population k_max range = [{population[-1][1]}, {population[0][1]}]")
    
    for iteration in range(1, max_iterations + 1):
        # Local refinement of top candidates
        refined_population = []
        for spheres, kmax in population[:population_size // 2]:
            current = spheres
            current_kmax = kmax
            
            for _ in range(3):  # 3 refinement steps
                magnitude_local = magnitude * (0.5 if best_kmax >= N - 2 else 1.0)
                refined = perturb_spheres(current, magnitude_local)
                refined_kmax = calculate_kmax(refined)
                
                if refined_kmax >= current_kmax:
                    current = refined
                    current_kmax = refined_kmax
            
            refined_population.append((current, current_kmax))
        
        # Breeding: cross over sphere parameters
        offspring = []
        for _ in range(population_size - len(refined_population)):
            parent1, _ = random.choice(population[:population_size // 2])
            parent2, _ = random.choice(population[:population_size // 2])
            
            # Cross over: blend parent parameters
            child_spheres = []
            for p1, p2 in zip(parent1, parent2):
                alpha = random.random()
                new_center = alpha * p1.center + (1 - alpha) * p2.center
                new_radius = alpha * p1.radius + (1 - alpha) * p2.radius
                child_spheres.append(Sphere(p1.id, new_center, max(new_radius, 0.1)))
            
            # Mutation
            child_spheres = perturb_spheres(child_spheres, magnitude * 0.5)
            child_kmax = calculate_kmax(child_spheres)
            offspring.append((child_spheres, child_kmax))
        
        # Merge and select best
        population = refined_population + offspring
        population.sort(key=lambda x: x[1], reverse=True)
        
        new_best = population[0][1]
        if new_best > best_kmax:
            best_kmax = new_best
            print(f"Iteration {iteration}: NEW MAX k_max = {best_kmax}")
        
        if iteration % (max_iterations // 10) == 0:
            avg_kmax = np.mean([k for _, k in population])
            print(f"Iteration {iteration}: Best={best_kmax}, Avg={avg_kmax:.2f}")
    
    print("--- Search Complete ---")
    return best_kmax, population[0][0]


def scan_parameter_space_dense(param_ranges: dict, N: int, resolution: int = 20):
    """
    Traditional dense grid scan followed by local refinement.
    Good for visualizing the parameter landscape.
    
    Parameters
    ----------
    param_ranges : dict
        Dictionary mapping parameter names to (min, max) ranges
    N : int
        Number of spheres
    resolution : int
        Grid resolution per parameter (total points = resolution^len(params))
    """
    print(f"--- Starting Dense Parameter Space Scan (N={N}, resolution={resolution}) ---")
    
    param_names = list(param_ranges.keys())
    param_values = [np.linspace(param_ranges[p][0], param_ranges[p][1], resolution) 
                    for p in param_names]
    
    results = []
    total_points = np.prod([resolution] * len(param_names))
    count = 0
    
    for values in np.ndindex(tuple([resolution] * len(param_names))):
        count += 1
        params = {param_names[i]: param_values[i][values[i]] for i in range(len(param_names))}
        
        # Generate spheres with these parameters
        spheres = generate_parametric_spheres(N, params)
        kmax = calculate_kmax(spheres)
        
        results.append({**params, 'k_max': kmax})
        
        if count % max(1, total_points // 20) == 0:
            print(f"  Progress: {count}/{total_points} points, current k_max={kmax}")
    
    return results


def scan_parameter_space_adaptive(param_ranges: dict, N: int, 
                                  initial_resolution: int = 5, 
                                  refinement_passes: int = 3):
    """
    Adaptive scanning that starts coarse and refines around high-k_max regions.
    Much more efficient at finding narrow peaks.
    """
    print(f"--- Starting Adaptive Parameter Space Scan (N={N}) ---")
    
    param_names = list(param_ranges.keys())
    results = []
    
    current_ranges = param_ranges.copy()
    current_resolution = initial_resolution
    
    for pass_num in range(refinement_passes):
        print(f"\n  Pass {pass_num + 1}: Resolution = {current_resolution}")
        
        param_values = [np.linspace(current_ranges[p][0], current_ranges[p][1], current_resolution) 
                        for p in param_names]
        
        pass_results = []
        total_points = np.prod([current_resolution] * len(param_names))
        count = 0
        
        for values in np.ndindex(tuple([current_resolution] * len(param_names))):
            count += 1
            params = {param_names[i]: param_values[i][values[i]] for i in range(len(param_names))}
            
            spheres = generate_parametric_spheres(N, params)
            kmax = calculate_kmax(spheres)
            
            pass_results.append({**params, 'k_max': kmax, 'pass': pass_num})
            
            if count % max(1, total_points // 10) == 0:
                print(f"    Progress: {count}/{total_points}, current k_max={kmax}")
        
        # Find the point with highest k_max and its neighborhood
        best_idx = np.argmax([r['k_max'] for r in pass_results])
        best_result = pass_results[best_idx]
        
        results.extend(pass_results)
        
        # Refine around the best point
        if pass_num < refinement_passes - 1:
            margin = 0.2  # 20% refinement margin
            for p in param_names:
                span = current_ranges[p][1] - current_ranges[p][0]
                margin_size = span * margin
                current_ranges[p] = (
                    max(param_ranges[p][0], best_result[p] - margin_size),
                    min(param_ranges[p][1], best_result[p] + margin_size)
                )
            
            print(f"  Refining around k_max={best_result['k_max']}")
            print(f"  New ranges: {current_ranges}")
            current_resolution = initial_resolution * 2  # Increase resolution for next pass
    
    return results


def generate_parametric_spheres(N: int, params: dict) -> List[Sphere]:
    """
    Generate N spheres based on a parameter dictionary.
    This is a placeholder - customize based on your specific parameterization.
    """
    # Example: params could be {'lattice_param': 4.2, 'offset_x': 0.1, 'radius': 2.5}
    # Implement your specific sphere generation here
    spheres = []
    for i in range(N):
        # Simple default: arrange spheres in a pattern based on params
        angle = 2 * np.pi * i / N
        center = [
            params.get('center_x', 0) + params.get('radius_factor', 2.0) * np.cos(angle),
            params.get('center_y', 0) + params.get('radius_factor', 2.0) * np.sin(angle),
            params.get('center_z', 0) + 0.5 * i
        ]
        radius = params.get('radius', 2.0) + params.get('radius_var', 0.0) * np.sin(angle)
        spheres.append(Sphere(i, center, radius))
    return spheres


# --- Example Usage ---
if __name__ == "__main__":
    N_SPHERES = 6
    
    print("=" * 70)
    print("RADICAL CENTER K_MAX: ENHANCED SEARCH STRATEGIES")
    print("=" * 70)
    print()
    
    # Test 1: Basic vs Adaptive
    print("\n1. BASIC HILL-CLIMBING SEARCH")
    print("-" * 70)
    final_kmax_basic, config_basic = randomized_search_basic(
        N_SPHERES, 
        max_iterations=1000, 
        magnitude=1e-4
    )
    print(f"Final k_max: {final_kmax_basic}\n")
    
    print("\n2. ADAPTIVE SEARCH (Recommended)")
    print("-" * 70)
    final_kmax_adaptive, config_adaptive = randomized_search_adaptive(
        N_SPHERES, 
        max_iterations=1000, 
        magnitude=1e-4
    )
    print(f"Final k_max: {final_kmax_adaptive}\n")
    
    print("\n3. GENETIC ALGORITHM SEARCH")
    print("-" * 70)
    final_kmax_genetic, config_genetic = randomized_search_genetic(
        N_SPHERES, 
        max_iterations=500, 
        magnitude=1e-4,
        population_size=8
    )
    print(f"Final k_max: {final_kmax_genetic}\n")
    
    print("\n" + "=" * 70)
    print(f"COMPARISON: Basic={final_kmax_basic}, Adaptive={final_kmax_adaptive}, Genetic={final_kmax_genetic}")
    print("=" * 70)
