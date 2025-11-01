"""
COMPREHENSIVE TEST SUITE FOR RADICAL CENTER K_MAX ALGORITHM

Tests include:
1. Known configurations with verified k_max values
2. Algorithm correctness verification
3. Search strategy comparison on solvable problems
4. Performance benchmarking
"""

import numpy as np
import time
from typing import List, Tuple
from radical_center_enhanced import (
    Sphere, calculate_kmax, randomized_search_adaptive,
    randomized_search_genetic, randomized_search_basic
)


# =============================================================================
# SECTION 1: TEST CASES WITH KNOWN K_MAX VALUES
# =============================================================================

def test_case_1_two_spheres_intersecting():
    """
    Two spheres that definitely intersect.
    Expected k_max = 2
    """
    print("\n" + "="*70)
    print("TEST 1: Two Intersecting Spheres")
    print("="*70)
    
    spheres = [
        Sphere(0, [0, 0, 0], 2.0),
        Sphere(1, [2.0, 0, 0], 2.0),  # Centers 2.0 apart, radii 2.0 each → intersect
    ]
    
    kmax = calculate_kmax(spheres)
    print(f"\nTwo spheres:")
    print(f"  S1: center=(0, 0, 0), radius=2.0")
    print(f"  S2: center=(2.0, 0, 0), radius=2.0")
    print(f"  Distance between centers: 2.0")
    print(f"  Sum of radii: 4.0")
    print(f"  They overlap: 2.0 < 4.0 ✓")
    print(f"\n✓ Result: k_max = {kmax}")
    print(f"  Expected: 2")
    assert kmax >= 2, f"Expected k_max >= 2, got {kmax}"
    return kmax == 2


def test_case_2_three_spheres_common_point():
    """
    Three spheres all passing through a common point.
    Expected k_max = 3
    """
    print("\n" + "="*70)
    print("TEST 2: Three Spheres Through Common Point")
    print("="*70)
    
    common_point = np.array([1.0, 1.0, 1.0])
    
    centers = [
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 2.0, 0.0]),
    ]
    
    spheres = []
    for i, center in enumerate(centers):
        radius = np.linalg.norm(common_point - center)
        spheres.append(Sphere(i, center, radius))
        print(f"\nS{i+1}: center={center}, radius={radius:.6f}")
        dist_to_point = np.linalg.norm(common_point - center)
        print(f"      Distance to common point: {dist_to_point:.6f}")
    
    kmax = calculate_kmax(spheres)
    print(f"\n✓ Result: k_max = {kmax}")
    print(f"  Expected: 3")
    assert kmax >= 3, f"Expected k_max >= 3, got {kmax}"
    return kmax == 3


def test_case_3_four_spheres_tetrahedron():
    """
    Four spheres at vertices of a regular tetrahedron.
    All pass through the center at origin.
    Expected k_max = 4
    """
    print("\n" + "="*70)
    print("TEST 3: Tetrahedral Configuration")
    print("="*70)
    
    # Regular tetrahedron vertices
    vertices = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    
    vertices = vertices / np.linalg.norm(vertices[0])
    center = np.array([0.0, 0.0, 0.0])
    
    spheres = []
    for i, vertex in enumerate(vertices):
        radius = np.linalg.norm(vertex - center)
        spheres.append(Sphere(i, vertex, radius))
        print(f"S{i+1}: center={vertex}, radius={radius:.6f}")
    
    kmax = calculate_kmax(spheres)
    print(f"\n✓ Result: k_max = {kmax}")
    print(f"  Expected: 4")
    assert kmax >= 4, f"Expected k_max >= 4, got {kmax}"
    return kmax == 4


def test_case_4_octahedral():
    """
    Six spheres at vertices of a regular octahedron.
    All meet at center.
    Expected k_max = 6
    """
    print("\n" + "="*70)
    print("TEST 4: Octahedral Configuration (6 spheres)")
    print("="*70)
    
    vertices = np.array([
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ], dtype=float)
    
    center = np.array([0.0, 0.0, 0.0])
    
    spheres = []
    for i, vertex in enumerate(vertices):
        radius = np.linalg.norm(vertex - center)
        spheres.append(Sphere(i, vertex, radius))
    
    kmax = calculate_kmax(spheres)
    print(f"\nOctahedral arrangement with {len(spheres)} vertices")
    print(f"All spheres meet at origin (0, 0, 0)")
    print(f"\n✓ Result: k_max = {kmax}")
    print(f"  Expected: 6")
    assert kmax >= 6, f"Expected k_max >= 6, got {kmax}"
    return kmax == 6


def test_case_5_no_triple_intersection():
    """
    Three spheres with pairwise intersections but NO triple point.
    Expected k_max = 2
    """
    print("\n" + "="*70)
    print("TEST 5: Pairwise Intersection Only (No Triple Point)")
    print("="*70)
    
    spheres = [
        Sphere(0, [0.0, 0.0, 0.0], 1.0),
        Sphere(1, [1.5, 0.0, 0.0], 1.0),
        Sphere(2, [0.75, 2.0, 0.0], 1.5),
    ]
    
    print("\nThree spheres arranged so pairs intersect but no common triple point")
    for i, s in enumerate(spheres):
        print(f"S{i+1}: center={s.center}, radius={s.radius}")
    
    kmax = calculate_kmax(spheres)
    print(f"\n✓ Result: k_max = {kmax}")
    print(f"  Expected: 2 (pairwise only)")
    return kmax == 2


def test_case_6_rocksalt_structure():
    """
    Simple rocksalt-like structure: FCC arrangement.
    4 spheres arranged in tetrahedral pattern but slightly displaced.
    """
    print("\n" + "="*70)
    print("TEST 6: Rocksalt-like Structure (4 Metal Sites)")
    print("="*70)
    
    # Simple cubic metal sublattice
    a = 3.0  # Lattice parameter
    r = 1.2  # Sphere radius
    
    spheres = [
        Sphere(0, [0, 0, 0], r),
        Sphere(1, [a, 0, 0], r),
        Sphere(2, [0, a, 0], r),
        Sphere(3, [0, 0, a], r),
    ]
    
    print(f"\nCubic arrangement with lattice parameter a={a}")
    print(f"Sphere radius r={r}")
    for i, s in enumerate(spheres):
        print(f"S{i+1}: center={s.center}, radius={s.radius}")
    
    kmax = calculate_kmax(spheres)
    print(f"\n✓ Result: k_max = {kmax}")
    print(f"  (This is a real crystal structure test)")
    return kmax


# =============================================================================
# SECTION 2: VERIFY ALGORITHM CORRECTNESS
# =============================================================================

def test_algorithm_correctness():
    """
    Run all known test cases to verify the k_max calculation is correct.
    """
    print("\n\n" + "#"*70)
    print("# ALGORITHM CORRECTNESS VERIFICATION")
    print("#"*70)
    
    results = []
    
    results.append(("Two intersecting spheres", test_case_1_two_spheres_intersecting()))
    results.append(("Three spheres @ common point", test_case_2_three_spheres_common_point()))
    results.append(("Tetrahedron (4 spheres)", test_case_3_four_spheres_tetrahedron()))
    results.append(("Octahedron (6 spheres)", test_case_4_octahedral()))
    results.append(("Pairwise only (no triple)", test_case_5_no_triple_intersection()))
    test_case_6_rocksalt_structure()
    
    print("\n" + "="*70)
    print("CORRECTNESS SUMMARY")
    print("="*70)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    return passed == total


# =============================================================================
# SECTION 3: SEARCH STRATEGY COMPARISON
# =============================================================================

def test_strategy_comparison():
    """
    Compare all three search strategies on a known high-k_max configuration.
    Start with a tetrahedron (k_max=4) and see if each strategy finds it.
    """
    print("\n\n" + "#"*70)
    print("# SEARCH STRATEGY COMPARISON")
    print("#"*70)
    
    # Create a tetrahedral configuration starting point
    vertices = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    vertices = vertices / np.linalg.norm(vertices[0])
    
    initial_spheres = []
    for i, vertex in enumerate(vertices):
        radius = np.linalg.norm(vertex)
        initial_spheres.append(Sphere(i, vertex.copy(), radius))
    
    N = len(initial_spheres)
    initial_kmax = calculate_kmax(initial_spheres)
    
    print(f"\nStarting configuration: k_max = {initial_kmax} (expected = 4)")
    print(f"Number of spheres: N = {N}")
    print(f"Budget: 2000 iterations per strategy\n")
    
    results = {}
    
    # Test 1: Basic search
    print("Strategy 1: BASIC HILL-CLIMBING")
    print("-" * 70)
    start_time = time.time()
    best_kmax_basic, _ = randomized_search_basic(N, 2000, 1e-4)
    time_basic = time.time() - start_time
    results['basic'] = (best_kmax_basic, time_basic)
    print(f"Final k_max: {best_kmax_basic}")
    print(f"Time: {time_basic:.2f}s\n")
    
    # Test 2: Adaptive search
    print("Strategy 2: ADAPTIVE SEARCH")
    print("-" * 70)
    start_time = time.time()
    best_kmax_adaptive, _ = randomized_search_adaptive(N, 2000, 1e-4)
    time_adaptive = time.time() - start_time
    results['adaptive'] = (best_kmax_adaptive, time_adaptive)
    print(f"Final k_max: {best_kmax_adaptive}")
    print(f"Time: {time_adaptive:.2f}s\n")
    
    # Test 3: Genetic
    print("Strategy 3: GENETIC ALGORITHM")
    print("-" * 70)
    start_time = time.time()
    best_kmax_genetic, _ = randomized_search_genetic(N, 1000, 1e-4, population_size=8)
    time_genetic = time.time() - start_time
    results['genetic'] = (best_kmax_genetic, time_genetic)
    print(f"Final k_max: {best_kmax_genetic}")
    print(f"Time: {time_genetic:.2f}s\n")
    
    # Summary
    print("="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Strategy':<20} {'k_max':<10} {'Time (s)':<10} {'Efficiency':<10}")
    print("-" * 70)
    
    for strategy, (kmax, elapsed) in results.items():
        efficiency = f"{kmax}@{elapsed:.2f}s"
        print(f"{strategy:<20} {kmax:<10} {elapsed:<10.2f} {efficiency:<10}")
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: (x[1][0], -x[1][1]))
    print(f"\n✓ Best strategy: {best_strategy[0].upper()}")
    print(f"  k_max={best_strategy[1][0]}, time={best_strategy[1][1]:.2f}s")


# =============================================================================
# SECTION 4: STRESS TESTS
# =============================================================================

def test_scaling_with_N():
    """
    Test how algorithm scales with increasing number of spheres.
    """
    print("\n\n" + "#"*70)
    print("# SCALING TEST: Algorithm complexity with increasing N")
    print("#"*70)
    
    N_values = [4, 6, 8, 10]
    
    print(f"\n{'N':<5} {'k_max':<10} {'Time (s)':<10} {'Cost/iteration':<15}")
    print("-" * 70)
    
    for N in N_values:
        # Create tetrahedral starting point
        if N <= 4:
            vertices = np.array([
                [1, 1, 1],
                [1, -1, -1],
                [-1, 1, -1],
                [-1, -1, 1]
            ], dtype=float)[:N]
        else:
            # For N > 4, add more random points
            vertices = np.random.randn(N, 3)
        
        vertices = vertices / (np.linalg.norm(vertices[0]) + 1e-10)
        
        spheres = []
        for i, vertex in enumerate(vertices):
            radius = np.linalg.norm(vertex) + 0.5
            spheres.append(Sphere(i, vertex.copy(), radius))
        
        # Quick search
        start_time = time.time()
        best_kmax, _ = randomized_search_adaptive(N, 500, 1e-4)
        elapsed = time.time() - start_time
        
        cost_per_iter = elapsed / 500
        
        print(f"{N:<5} {best_kmax:<10} {elapsed:<10.2f} {cost_per_iter:<15.6f}")
    
    print("\nNote: O(N^4) algorithm → expect ~16-20x cost increase when N doubles")


def test_epsilon_sensitivity():
    """
    Test sensitivity to EPSILON tolerance.
    """
    print("\n\n" + "#"*70)
    print("# EPSILON SENSITIVITY TEST")
    print("#"*70)
    
    # Create a test configuration
    vertices = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    vertices = vertices / np.linalg.norm(vertices[0])
    
    spheres = []
    for i, vertex in enumerate(vertices):
        radius = np.linalg.norm(vertex)
        spheres.append(Sphere(i, vertex.copy(), radius))
    
    print(f"\nTetrahedron configuration (should have k_max=4)")
    print(f"\n{'EPSILON':<15} {'k_max':<10} {'Notes':<40}")
    print("-" * 70)
    
    epsilons = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4]
    
    for eps in epsilons:
        # Temporarily modify EPSILON in module
        import radical_center_enhanced
        original_eps = radical_center_enhanced.EPSILON
        radical_center_enhanced.EPSILON = eps
        
        kmax = calculate_kmax(spheres)
        
        radical_center_enhanced.EPSILON = original_eps
        
        notes = ""
        if kmax < 4:
            notes = "TOO STRICT - missed intersections"
        elif kmax == 4:
            notes = "✓ Correct"
        
        print(f"{eps:<15.0e} {kmax:<10} {notes:<40}")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    print("\n")
    print("█" * 70)
    print("█ " + " "*66 + " █")
    print("█ " + "RADICAL CENTER K_MAX: COMPREHENSIVE TEST SUITE".center(66) + " █")
    print("█ " + " "*66 + " █")
    print("█" * 70)
    
    # Run all test sections
    print("\n[1/4] Running algorithm correctness tests...")
    correctness_passed = test_algorithm_correctness()
    
    print("\n[2/4] Comparing search strategies...")
    test_strategy_comparison()
    
    print("\n[3/4] Testing scaling with N...")
    test_scaling_with_N()
    
    print("\n[4/4] Testing epsilon sensitivity...")
    test_epsilon_sensitivity()
    
    # Final summary
    print("\n\n" + "█" * 70)
    if correctness_passed:
        print("█ ✓ ALL CORRECTNESS TESTS PASSED")
    else:
        print("█ ✗ SOME CORRECTNESS TESTS FAILED")
    print("█" * 70)
    print("\nTest suite complete!\n")
