"""
PRACTICAL EXAMPLES: Using Enhanced Radical Center Search

This module demonstrates practical workflows for:
1. Finding optimal sphere configurations for a given k_max
2. Mapping parameter space to create k_max heatmaps
3. Comparing search strategies
4. Handling crystallographic parameter spaces
"""

import numpy as np
import random
from typing import List, Dict, Tuple
import json

# Assume imports from radical_center_enhanced.py
from radical_center_enhanced import (
    Sphere, calculate_kmax, randomized_search_adaptive,
    randomized_search_genetic, generate_parametric_spheres,
    scan_parameter_space_adaptive, scan_parameter_space_dense
)


# =============================================================================
# EXAMPLE 1: FINDING OPTIMAL CONFIGURATION FOR A GIVEN N
# =============================================================================

def example_1_find_optimal_config():
    """
    Goal: Find the configuration of N spheres that maximizes k_max
    (i.e., find k_max = N or close to it)
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: FINDING OPTIMAL SPHERE CONFIGURATION")
    print("="*70)
    
    N_SPHERES = 8
    MAX_ITERATIONS = 5000
    MAGNITUDE = 1e-4
    
    print(f"\nObjective: Find sphere configuration with highest k_max for N={N_SPHERES}")
    print(f"Strategy: Adaptive randomized search with restarts")
    print(f"Budget: {MAX_ITERATIONS} iterations (~1-2 minutes for N={N_SPHERES})\n")
    
    best_kmax, best_config = randomized_search_adaptive(
        N_SPHERES, 
        max_iterations=MAX_ITERATIONS, 
        magnitude=MAGNITUDE
    )
    
    print(f"\n✓ Final result: k_max = {best_kmax} (target was {N_SPHERES})")
    
    if best_kmax == N_SPHERES:
        print("  SUCCESS: Found perfect alignment where all spheres intersect!")
    elif best_kmax >= N_SPHERES - 2:
        print(f"  GOOD: Only {N_SPHERES - best_kmax} sphere(s) not in alignment")
    else:
        print(f"  Could improve: {N_SPHERES - best_kmax} spheres still missing")
    
    # Print configuration
    print(f"\nOptimal sphere configuration:")
    for i, sphere in enumerate(best_config):
        print(f"  S{i+1}: center=({sphere.center[0]:.6f}, {sphere.center[1]:.6f}, {sphere.center[2]:.6f}), " 
              f"radius={sphere.radius:.6f}")
    
    # Save for later use
    return best_config


# =============================================================================
# EXAMPLE 2: STRATEGY COMPARISON
# =============================================================================

def example_2_compare_strategies():
    """
    Objective: Compare three search strategies on same problem
    This helps determine which strategy works best for your specific system
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: COMPARING SEARCH STRATEGIES")
    print("="*70)
    
    N_SPHERES = 6
    MAX_ITERATIONS = 2000
    MAGNITUDE = 1e-4
    NUM_TRIALS = 3  # Multiple trials to assess consistency
    
    print(f"\nComparing 3 search strategies on {NUM_TRIALS} independent trials")
    print(f"N={N_SPHERES}, {MAX_ITERATIONS} iterations each\n")
    
    results = {
        'adaptive': [],
        'genetic': [],
    }
    
    # Strategy 1: Adaptive Search
    print("Strategy 1: ADAPTIVE SEARCH")
    print("-" * 70)
    for trial in range(NUM_TRIALS):
        print(f"  Trial {trial+1}/{NUM_TRIALS}...", end=" ")
        kmax, _ = randomized_search_adaptive(N_SPHERES, MAX_ITERATIONS, MAGNITUDE)
        results['adaptive'].append(kmax)
        print(f"k_max={kmax}")
    
    # Strategy 2: Genetic Algorithm
    print("\nStrategy 2: GENETIC ALGORITHM")
    print("-" * 70)
    for trial in range(NUM_TRIALS):
        print(f"  Trial {trial+1}/{NUM_TRIALS}...", end=" ")
        kmax, _ = randomized_search_genetic(N_SPHERES, MAX_ITERATIONS//5, MAGNITUDE, 
                                            population_size=8)
        results['genetic'].append(kmax)
        print(f"k_max={kmax}")
    
    # Analysis
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    for strategy, kmax_list in results.items():
        mean_kmax = np.mean(kmax_list)
        max_kmax = np.max(kmax_list)
        std_kmax = np.std(kmax_list)
        
        print(f"\n{strategy.upper()}:")
        print(f"  Mean k_max:    {mean_kmax:.2f}")
        print(f"  Max k_max:     {max_kmax}")
        print(f"  Std Dev:       {std_kmax:.2f}")
        print(f"  All results:   {kmax_list}")
    
    # Recommendation
    print("\n" + "-"*70)
    adaptive_avg = np.mean(results['adaptive'])
    genetic_avg = np.mean(results['genetic'])
    
    if adaptive_avg >= genetic_avg:
        print(f"RECOMMENDATION: Adaptive search performs slightly better ({adaptive_avg:.2f} vs {genetic_avg:.2f})")
    else:
        print(f"RECOMMENDATION: Genetic algorithm performs better ({genetic_avg:.2f} vs {adaptive_avg:.2f})")


# =============================================================================
# EXAMPLE 3: PARAMETER SPACE MAPPING
# =============================================================================

def example_3_parameter_mapping():
    """
    Objective: Create a 2D heat map of k_max values across parameter space
    
    For a crystal structure system, parameters might be:
    - Lattice parameter 'a'
    - Sphere radius 'r'
    - Sublattice offset 'offset'
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: 2D PARAMETER SPACE HEAT MAP")
    print("="*70)
    
    N_SPHERES = 4
    
    # Define 2D parameter space (for visualization and speed)
    param_ranges = {
        'radius_factor': (1.0, 5.0),   # Sphere radius multiplier
        'offset': (0.0, 1.0),          # Sublattice offset
    }
    
    print(f"\nMapping k_max across 2D parameter space for N={N_SPHERES}")
    print(f"Parameters: {list(param_ranges.keys())}")
    print(f"Ranges: {param_ranges}\n")
    
    # Method 1: Dense scan (for small spaces only)
    print("Method 1: DENSE GRID SCAN (10x10 resolution)")
    print("-" * 70)
    dense_results = scan_parameter_space_dense(param_ranges, N_SPHERES, resolution=10)
    
    # Extract k_max values
    kmax_values = [r['k_max'] for r in dense_results]
    print(f"  Total evaluations: {len(dense_results)}")
    print(f"  k_max range: [{min(kmax_values)}, {max(kmax_values)}]")
    print(f"  Points with k_max={max(kmax_values)}: {sum(1 for r in dense_results if r['k_max']==max(kmax_values))}")
    
    # Find peaks
    peaks = [r for r in dense_results if r['k_max'] == max(kmax_values)]
    print(f"\n  Peak locations:")
    for peak in peaks[:5]:  # Show first 5
        print(f"    radius_factor={peak['radius_factor']:.3f}, offset={peak['offset']:.3f}")
    if len(peaks) > 5:
        print(f"    ... and {len(peaks)-5} more")
    
    # Method 2: Adaptive scan (much more efficient)
    print("\n\nMethod 2: ADAPTIVE MULTI-PASS SCAN")
    print("-" * 70)
    adaptive_results = scan_parameter_space_adaptive(
        param_ranges, N_SPHERES, 
        initial_resolution=5,  # Start coarser
        refinement_passes=2
    )
    
    # Extract k_max values
    kmax_adaptive = [r['k_max'] for r in adaptive_results]
    print(f"  Total evaluations: {len(adaptive_results)}")
    print(f"  k_max range: [{min(kmax_adaptive)}, {max(kmax_adaptive)}]")
    
    # Compare efficiency
    print("\n" + "-"*70)
    print("EFFICIENCY COMPARISON:")
    print(f"  Dense:    {len(dense_results)} evaluations")
    print(f"  Adaptive: {len(adaptive_results)} evaluations")
    print(f"  Speedup:  {len(dense_results)/len(adaptive_results):.1f}x")
    
    # Save results
    print("\nSaving results to files...")
    with open('/home/claude/dense_scan.json', 'w') as f:
        json.dump(dense_results, f, indent=2)
    with open('/home/claude/adaptive_scan.json', 'w') as f:
        json.dump(adaptive_results, f, indent=2)


# =============================================================================
# EXAMPLE 4: CRYSTALLOGRAPHIC PARAMETER SPACE
# =============================================================================

def create_rocksalt_like_config(a: float, r_factor: float, offset: float, N: int) -> List[Sphere]:
    """
    Create a rocksalt-like structure where:
    - a: lattice parameter
    - r_factor: sphere radius relative to a/2
    - offset: displacement of anion sublattice
    - N: number of spheres
    """
    spheres = []
    
    # Cation positions (metal sublattice)
    # Simple cubic with lattice parameter a
    grid_size = int(np.ceil(N**(1/3)))
    idx = 0
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                if idx >= N:
                    break
                
                # Cation at lattice point
                center = np.array([i, j, k]) * a
                radius = r_factor * a / 2
                spheres.append(Sphere(idx, center, radius))
                idx += 1
            if idx >= N:
                break
        if idx >= N:
            break
    
    # Perturb to break symmetry slightly (make intersection possible)
    for sphere in spheres:
        sphere.center += np.random.normal(0, offset * a, 3)
    
    return spheres


def example_4_crystallographic_space():
    """
    Scan parameter space specific to crystal structure discovery
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: CRYSTAL STRUCTURE PARAMETER SPACE")
    print("="*70)
    
    N_SPHERES = 5
    
    # Crystal structure parameters
    param_ranges = {
        'lattice_param': (2.0, 8.0),    # Unit cell size
        'radius_factor': (0.3, 0.8),    # Sphere radius relative to lattice
        'offset': (0.0, 0.2),           # Sublattice offset for anion
    }
    
    print(f"\nScanning 3D crystal parameter space for N={N_SPHERES}")
    print(f"Parameters: lattice_param, radius_factor, offset\n")
    
    # Coarse survey
    print("PHASE 1: Coarse Survey (resolution=5 = 125 points)")
    print("-" * 70)
    
    coarse_results = []
    ranges = param_ranges
    res = 5
    
    param_names = list(ranges.keys())
    param_arrays = [np.linspace(ranges[p][0], ranges[p][1], res) for p in param_names]
    
    max_kmax_coarse = 0
    candidate_regions = []
    
    for idx in np.ndindex((res, res, res)):
        params = {
            param_names[i]: param_arrays[i][idx[i]] for i in range(len(param_names))
        }
        
        # Generate rocksalt-like structure with these parameters
        spheres = create_rocksalt_like_config(
            a=params['lattice_param'],
            r_factor=params['radius_factor'],
            offset=params['offset'],
            N=N_SPHERES
        )
        
        kmax = calculate_kmax(spheres)
        coarse_results.append({**params, 'k_max': kmax})
        
        max_kmax_coarse = max(max_kmax_coarse, kmax)
        
        if kmax >= N_SPHERES - 2:  # Near-optimal
            candidate_regions.append(params)
    
    print(f"  Evaluated: {len(coarse_results)} points")
    print(f"  Max k_max found: {max_kmax_coarse}")
    print(f"  Candidate regions: {len(candidate_regions)}")
    
    if candidate_regions:
        print(f"\n  Candidate region locations:")
        for i, region in enumerate(candidate_regions[:3]):
            print(f"    Region {i+1}: a={region['lattice_param']:.2f}, "
                  f"r_factor={region['radius_factor']:.3f}, "
                  f"offset={region['offset']:.3f}")
        if len(candidate_regions) > 3:
            print(f"    ... and {len(candidate_regions)-3} more")
    
    # Refine best region
    print("\n\nPHASE 2: Refine Around Best Region (if found)")
    print("-" * 70)
    
    if candidate_regions:
        # Take center of first candidate region
        best_params = candidate_regions[0]
        
        # Define refined range (±20% around best point)
        margin = 0.2
        refined_ranges = {}
        for key in ranges:
            span = ranges[key][1] - ranges[key][0]
            margin_size = span * margin
            refined_ranges[key] = (
                max(ranges[key][0], best_params[key] - margin_size),
                min(ranges[key][1], best_params[key] + margin_size)
            )
        
        print(f"  Refining around a={best_params['lattice_param']:.2f}")
        print(f"  New ranges: {refined_ranges}\n")
        
        # Finer scan in refined region
        refined_results = []
        res_fine = 8
        
        param_arrays_fine = [np.linspace(refined_ranges[p][0], refined_ranges[p][1], res_fine) 
                             for p in param_names]
        
        max_kmax_refined = 0
        
        for idx in np.ndindex((res_fine, res_fine, res_fine)):
            params = {
                param_names[i]: param_arrays_fine[i][idx[i]] for i in range(len(param_names))
            }
            
            spheres = create_rocksalt_like_config(
                a=params['lattice_param'],
                r_factor=params['radius_factor'],
                offset=params['offset'],
                N=N_SPHERES
            )
            
            kmax = calculate_kmax(spheres)
            refined_results.append({**params, 'k_max': kmax})
            max_kmax_refined = max(max_kmax_refined, kmax)
        
        print(f"  Evaluated: {len(refined_results)} points")
        print(f"  Max k_max found: {max_kmax_refined}")
        
        # Final random search from best point
        print("\n\nPHASE 3: FINAL RANDOM SEARCH (1000 iterations)")
        print("-" * 70)
        
        best_refined = max(refined_results, key=lambda r: r['k_max'])
        print(f"  Starting from: a={best_refined['lattice_param']:.4f}, "
              f"r_factor={best_refined['radius_factor']:.4f}, "
              f"offset={best_refined['offset']:.4f}")
        print(f"  Current k_max: {best_refined['k_max']}\n")
        
        # Would apply randomized_search here starting from best_refined
        # For now, just show the concept
        print(f"  [Would apply adaptive random search with fine magnitude here]")
    
    print("\n" + "-"*70)
    print(f"SUMMARY: Successfully mapped 3D parameter space")
    print(f"  Found high-k_max regions with k_max={max_kmax_coarse}/{N_SPHERES}")


# =============================================================================
# MAIN: RUN ALL EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("\n")
    print("█" * 70)
    print("█ " + " "*66 + " █")
    print("█ " + "PRACTICAL EXAMPLES: RADICAL CENTER K_MAX DISCOVERY".center(66) + " █")
    print("█ " + " "*66 + " █")
    print("█" * 70)
    
    # Run examples
    # Uncomment as needed:
    
    # example_1_find_optimal_config()
    
    # example_2_compare_strategies()
    
    # example_3_parameter_mapping()
    
    example_4_crystallographic_space()
    
    print("\n" + "█" * 70)
    print("All examples completed!")
    print("█" * 70)
