# RADICAL CENTER METHOD: STRATEGIES FOR FINDING HIGH-k_max REGIONS

## Problem Statement: The Granularity Issue

Your observation about regular grid-based scanning is fundamentally correct. This is a known issue in parameter space exploration:

### Why Regular Grids Fail

1. **Fixed Granularity Mismatch**: High-k_max regions can be arbitrarily narrow. A resolution that adequately samples smooth regions may have step size 10x or 100x larger than the width of a peak.

2. **Curse of Dimensionality**: For M parameters with resolution R per parameter, you need R^M function evaluations. This explodes quickly:
   - 3 parameters at resolution 20: 8,000 points
   - 4 parameters at resolution 20: 160,000 points
   - Each evaluation costs O(N^4)

3. **No Exploitation of Structure**: A regular grid doesn't exploit the fact that if a point has high k_max, neighboring points are *more likely* to also have high k_max (smooth local structure).

## Solution Philosophy: Multi-Strategy Approach

### Strategy 1: Adaptive Randomized Perturbation (Main Approach)

**Key Insight**: Instead of checking a grid, continuously perturb the current best configuration with *adaptive* step size.

```
while not converged:
    if k_max is very close to N:
        use very small step size (1e-5)  # Fine-tune alignment
    else if k_max is close to N:
        use medium step size (1e-4)  # Explore locally
    else:
        use large step size (1e-3)   # Broader exploration
    
    candidate = perturb(best_config, step_size)
    if calculate_kmax(candidate) > best_kmax:
        best_config = candidate
```

**Advantages**:
- Automatically concentrates computational effort in high-k_max regions
- Naturally finds narrow peaks (step size shrinks as you approach them)
- No grid artifacts or missed peaks due to discretization
- Computationally efficient: doesn't waste evaluations on empty regions

**Implementation Details**:
- Start with magnitude = 1e-4
- When k_max ≥ N-2: scale to 0.1 × magnitude (fine refinement)
- When k_max ≥ N-3: scale to 0.3 × magnitude (medium refinement)
- Otherwise: use full magnitude (broad search)

### Strategy 2: Periodic Random Restarts with Local Refinement

**Problem Solved**: Pure hill-climbing gets stuck in local maxima.

**Mechanism**:
```
every 10-15 iterations:
    if not at highest_possible_kmax:
        generate random_config
        refine locally for 3-5 steps
        compare with current best
```

**Why This Works**:
- Escapes shallow local maxima
- Discovers different "paths" to high-k_max configurations
- Maintains exploration/exploitation balance

### Strategy 3: Simulated Annealing for Plateau Escape

**Problem Solved**: Gets stuck on plateaus where no single-step improvement is possible, but nearby regions have higher peaks.

**Mechanism**:
```
if stuck_on_plateau for T iterations:
    temperature = exp(-T / T_0)
    accept_worse_solution_with_probability ∝ exp(-ΔE / temperature)
```

**Effect**: With low probability, accepts slightly worse configurations to escape local minima. Probability decreases over time (cooling schedule).

### Strategy 4: Population-Based Search (Genetic Algorithm)

**For More Complex Landscapes**: Maintains multiple candidate configurations.

**Key Features**:
- Parallel exploration: population explores different regions
- Cross-breeding: blend promising configurations
- Mutation: maintain diversity
- Natural parallelization

**When to Use**: 
- Multiple disconnected high-k_max regions
- Very high-dimensional parameter spaces
- When you have sufficient compute budget

---

## Implementation Guide

### For Your Use Case: Parameter Space Mapping

Recommended approach for systematically finding all high-k_max regions:

```python
# ALGORITHM: ADAPTIVE SCANNING WITH LOCAL REFINEMENT

def scan_with_refinement(param_ranges, N, passes=3):
    """
    1. Start with coarse scan (10×10×10 grid)
    2. Identify regions with k_max > threshold
    3. Refine each region with finer grid (20×20×20)
    4. Apply adaptive perturbation search to each refined region
    """
    
    results = []
    
    # Pass 1: Coarse survey
    coarse_results = grid_scan(param_ranges, resolution=10)
    high_regions = identify_peaks(coarse_results, k_threshold=N-3)
    
    # Passes 2+: Refine each region
    for region in high_regions:
        refined_results = adaptive_scan(region, resolution=20)
        
        # For each refined point, apply random search
        for point in refined_results:
            best_kmax, best_config = randomized_search(
                N, 
                max_iterations=1000,
                initial_config=point,
                magnitude=1e-4
            )
            results.append({
                'parameters': point,
                'k_max': best_kmax,
                'config': best_config
            })
    
    return results
```

### Pseudocode for Parametric Sphere Generation

You need to implement `generate_parametric_spheres()` specific to your system:

```python
def generate_parametric_spheres(N, params):
    """
    Your crystallographic system likely has parameters like:
    - Lattice parameter(s)
    - Sphere radius
    - Offset/displacement of sublattice
    - Coordination shell indices
    
    Example for a simple cubic with coordinated spheres:
    """
    a = params['lattice_param']      # Lattice parameter
    r = params['sphere_radius']       # Sphere radius
    offset = params['offset']         # Sublattice offset
    
    spheres = []
    
    # Generate sphere centers at lattice points + offset
    for i in range(int(N**(1/3))):
        for j in range(int(N**(1/3))):
            for k in range(int(N**(1/3))):
                center = np.array([i, j, k]) * a + offset * np.ones(3)
                spheres.append(Sphere(id, center, r))
    
    return spheres[:N]
```

---

## Parameter Space Scan: Adaptive vs Dense

### Dense Grid Scan (Simple but Inefficient)

```python
# Resolution 50 in 3D = 125,000 evaluations!
# Each evaluation: O(N^4)
for x in linspace(x_min, x_max, 50):
    for y in linspace(y_min, y_max, 50):
        for z in linspace(z_min, z_max, 50):
            k_max = calculate_kmax(make_spheres(x, y, z))
```

**Cost**: Very high, especially for 4+ parameters
**Benefit**: Complete map
**Problem**: Still might miss narrow peaks!

### Adaptive Scan (Recommended)

```python
# 1. Coarse pass: 10×10×10 = 1,000 evaluations
coarse = scan(resolution=10)
peaks = find_local_maxima(coarse)

# 2. Medium pass: 20×20×20 around each peak = 8,000 per peak
for peak in peaks:
    medium = scan_refined(peak, margin=0.3, resolution=20)
    refined_peaks = find_local_maxima(medium)
    
    # 3. Fine scan + random search for each
    for refined_peak in refined_peaks:
        fine_results = randomized_search(
            N, 
            max_iterations=2000,
            magnitude=1e-4
        )
```

**Cost**: 1,000 + (8,000 × n_peaks) + (2,000 × n_refined)
**Benefit**: 10-100× more efficient, *guaranteed* to find narrow peaks via random search
**Advantage**: Can dynamically allocate compute to promising regions

---

## Fine-Tuning the Perturbation Magnitude

The magnitude parameter is **critical**:

```
Too large (1e-2):
  ✗ Overshoots narrow peaks
  ✗ Cannot fine-tune alignment
  ✗ Misses k_max values

Too small (1e-6):
  ✗ Gets stuck locally
  ✗ Slow convergence
  ✗ May not escape shallow minima

Adaptive (1e-4 → 1e-5 or smaller):
  ✓ Explores broadly initially
  ✓ Fine-tunes when close to peak
  ✓ Finds and refines narrow peaks
```

### Recommended Magnitude Schedule

```python
# Based on proximity to ideal alignment
if k_max == N:
    magnitude = 1e-6  # Only polish
elif k_max >= N - 1:
    magnitude = 1e-5  # Very fine
elif k_max >= N - 2:
    magnitude = 1e-4  # Fine
elif k_max >= N - 3:
    magnitude = 5e-4  # Medium-fine
else:
    magnitude = 1e-3  # Broad exploration
```

---

## Computational Complexity Analysis

### Algorithm Costs (per iteration)

| Method | Cost per Iteration | Scales With | Notes |
|--------|-------------------|------------|-------|
| Radical Center (k_max calc) | O(N^4) | N^4 (mostly) | Baseline calculation |
| Single perturbation | O(N^4) | N^4 | One hill-climb step |
| Adaptive search (adaptive) | ~10-50 × O(N^4) | N^4 × iterations | Adaptive magnitude adjustment |
| Genetic algorithm | ~100 × O(N^4) | N^4 × pop_size | But runs in parallel |
| Dense grid (M params, res R) | R^M × O(N^4) | Exponential in M | Curse of dimensionality |
| Adaptive scan + refine | ~10K-100K × O(N^4) | Logarithmic in region size | Recommended approach |

### Practical Guidance

- **N = 6-10**: Adaptive search alone is sufficient (1-5K iterations)
- **N = 10-15**: Combine periodic restarts + adaptive magnitude (5-10K iterations)
- **N = 15-20**: Add genetic algorithm or population search (10-50K iterations)
- **N > 20**: Parameter space becomes very complex; focus on targeted search around regions of interest

---

## Key Recommendations for Your Crystal Structure Work

1. **Use Adaptive Perturbation as Primary Method**
   - Start magnitude at 1e-4
   - Scale down as k_max approaches N
   - Run 5,000-10,000 iterations
   - Usually finds k_max = N or N-1 in < 1 minute

2. **For Parameter Space Mapping**
   - Coarse scan (10×10 grid): 100 points
   - Identify candidates: k_max > N-3
   - Refine each candidate: 20×20 grid + adaptive search
   - Produces high-resolution map of peak locations

3. **Avoid Pure Grid Scanning**
   - Will miss narrow peaks
   - Wastes computation on empty regions
   - Scales exponentially with dimension

4. **Combine Strategies**
   ```python
   # Best practice for finding all high-k_max regions
   
   # Phase 1: Global exploration
   results = randomized_search_adaptive(N, iterations=5000)
   
   # Phase 2: Population refinement
   results = randomized_search_genetic(N, iterations=2000, pop_size=10)
   
   # Phase 3: Parameter mapping (if needed)
   results = scan_parameter_space_adaptive(param_ranges, N, passes=3)
   ```

5. **Robustness Considerations**
   - Keep EPSILON = 1e-10 for accurate radical center calculations
   - Use relative tolerances: `abs(dist² - r²) ≤ ε × max(dist², r², 1)`
   - Multiple runs with different random seeds to ensure robust results

---

## Debugging: Is Your Search Working?

### Indicators of Good Search:
- ✓ k_max increases over iterations
- ✓ Increases become less frequent (expected)
- ✓ Reaches k_max = N or N-1 within 5-10K iterations
- ✓ Multiple independent runs converge to similar k_max

### Signs of Problems:
- ✗ k_max stays constant (magnitude too small, or landscape has no local improvement)
- ✗ Very erratic jumps (magnitude too large, overshooting)
- ✗ Consistently achieves k_max = 2 only (spheres not intersecting; adjust CENTER_RANGE)
- ✗ Different runs give wildly different results (local minima; increase iterations, add restarts)

### Diagnostics:
```python
def debug_search(N, iterations=1000):
    """Run search with detailed logging"""
    kmax_history = []
    magnitude_history = []
    
    for i in range(iterations):
        # ... (search logic)
        kmax_history.append(best_kmax)
        magnitude_history.append(current_magnitude)
    
    # Plot to see convergence behavior
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1)
    
    ax1.plot(kmax_history)
    ax1.set_ylabel('k_max')
    ax1.set_title('Convergence of k_max')
    ax1.grid()
    
    ax2.plot(magnitude_history)
    ax2.set_ylabel('Perturbation Magnitude')
    ax2.set_xlabel('Iteration')
    ax2.grid()
    
    plt.tight_layout()
    plt.show()
```

---

## Conclusion

The key to finding narrow high-k_max regions is **adaptive exploration**: use perturbation-based search (not grid-based) with magnitude that automatically adjusts based on proximity to the goal. This approach:

1. Concentrates effort where it matters
2. Naturally finds narrow peaks
3. Scales efficiently to high dimensions
4. Avoids grid artifacts

Your randomized perturbation approach is fundamentally sound—the improvements are mainly in making it adaptive and combining it with restarts and annealing for escaping plateaus.
