import streamlit as st
import numpy as np
from radical_center_enhanced import Sphere, calculate_kmax
import plotly.graph_objects as go

st.set_page_config(page_title="Radical Center k_max Calculator", layout="wide")

st.title("🔬 Radical Center k_max Calculator")
st.markdown("**Analyze sphere intersection orders in crystal lattices**")

# Sidebar for configuration
st.sidebar.header("Configuration")

lattice_type = st.sidebar.selectbox(
    "Lattice Type",
    ["FCC (Face-Centered Cubic)", "BCC (Body-Centered Cubic)", "SC (Simple Cubic)"]
)

system_size = st.sidebar.selectbox(
    "System Size",
    ["2×2×2", "3×3×3", "4×4×4"]
)

a = st.sidebar.slider(
    "Lattice Parameter (a) in Ångströms",
    min_value=0.5,
    max_value=3.0,
    value=1.0,
    step=0.1
)

r = st.sidebar.number_input(
    "Sphere Radius (r) in Ångströms",
    min_value=0.1,
    max_value=10.0,
    value=0.5,
    step=0.0001,
    format="%.4f"
)

epsilon = st.sidebar.number_input(
    "Epsilon (tolerance)",
    min_value=1e-12,
    max_value=1e-3,
    value=1e-10,
    step=1e-11,
    format="%.2e"
)

# Helper function to generate lattice
def generate_fcc_lattice(size_multiplier, a):
    """Generate FCC lattice"""
    fcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ])
    
    positions = []
    for i in range(size_multiplier):
        for j in range(size_multiplier):
            for k in range(size_multiplier):
                cell_origin = np.array([i, j, k]) * a
                for basis_pos in fcc_basis:
                    pos = cell_origin + basis_pos * a
                    positions.append(pos)
    
    # Remove duplicates
    unique_positions = []
    for pos in positions:
        is_duplicate = False
        for unique_pos in unique_positions:
            if np.linalg.norm(pos - unique_pos) < 1e-10:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_positions.append(pos)
    
    return np.array(unique_positions)

def generate_bcc_lattice(size_multiplier, a):
    """Generate BCC lattice"""
    bcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0.5]
    ])
    
    positions = []
    for i in range(size_multiplier):
        for j in range(size_multiplier):
            for k in range(size_multiplier):
                cell_origin = np.array([i, j, k]) * a
                for basis_pos in bcc_basis:
                    pos = cell_origin + basis_pos * a
                    positions.append(pos)
    
    unique_positions = []
    for pos in positions:
        is_duplicate = False
        for unique_pos in unique_positions:
            if np.linalg.norm(pos - unique_pos) < 1e-10:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_positions.append(pos)
    
    return np.array(unique_positions)

def generate_sc_lattice(size_multiplier, a):
    """Generate Simple Cubic lattice"""
    positions = []
    for i in range(size_multiplier):
        for j in range(size_multiplier):
            for k in range(size_multiplier):
                positions.append(np.array([i, j, k]) * a)
    
    return np.array(positions)

# Generate lattice based on selection
size_map = {"2×2×2": 2, "3×3×3": 3, "4×4×4": 4}
size_multiplier = size_map[system_size]

if "FCC" in lattice_type:
    positions = generate_fcc_lattice(size_multiplier, a)
    lattice_name = "FCC"
elif "BCC" in lattice_type:
    positions = generate_bcc_lattice(size_multiplier, a)
    lattice_name = "BCC"
else:
    positions = generate_sc_lattice(size_multiplier, a)
    lattice_name = "SC"

# Create sphere objects
spheres = []
for idx, pos in enumerate(positions):
    spheres.append(Sphere(idx, pos, r))

import radical_center_enhanced
radical_center_enhanced.EPSILON = epsilon

# Calculate k_max
with st.spinner("Calculating k_max..."):
    kmax = calculate_kmax(spheres)

# Main display
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Lattice Type", lattice_name)
with col2:
    st.metric("System Size", system_size)
with col3:
    st.metric("Number of Spheres", len(spheres))

st.markdown("---")

# Big result display
col1, col2 = st.columns(2)

with col1:
    st.metric("Lattice Parameter (a)", f"{a:.2f} Å")
    st.metric("Sphere Radius (r)", f"{r:.2f} Å")
    st.metric("Sum of Radii (2r)", f"{2*r:.2f} Å")

with col2:
    # Highlight the result
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 10px;
        text-align: center;
        color: white;
    ">
        <h2 style="margin: 0; font-size: 2.5em;">k_max = {kmax}</h2>
        <p style="margin: 10px 0 0 0; font-size: 1.1em;">Maximum Intersection Order</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Analysis section
st.subheader("📊 Geometric Analysis")

col1, col2 = st.columns(2)

with col1:
    # Calculate nearest neighbor distance
    if len(spheres) > 1:
        distances = []
        for i in range(len(spheres)):
            for j in range(i+1, len(spheres)):
                dist = np.linalg.norm(spheres[i].center - spheres[j].center)
                distances.append(dist)
        
        distances = sorted(set(np.round(distances, 6)))
        
        st.write("**Nearest Neighbor Distances:**")
        for idx, d in enumerate(distances[:5]):
            sum_radii = 2 * r
            if d < sum_radii:
                status = "✓ OVERLAPPING"
                overlap = sum_radii - d
                st.write(f"{idx+1}. d = {d:.4f} Å (2r={sum_radii:.4f}) {status} (overlap: {overlap:.4f})")
            else:
                st.write(f"{idx+1}. d = {d:.4f} Å (2r={sum_radii:.4f}) Separated")

with col2:
    # Intersection statistics
    intersection_count = 0
    for i in range(len(spheres)):
        for j in range(i+1, len(spheres)):
            dist = np.linalg.norm(spheres[i].center - spheres[j].center)
            if dist <= 2*r + 1e-10:
                intersection_count += 1
    
    total_pairs = len(spheres) * (len(spheres) - 1) // 2
    intersection_ratio = 100.0 * intersection_count / total_pairs if total_pairs > 0 else 0
    
    st.write("**Intersection Statistics:**")
    st.write(f"- Intersecting pairs: {intersection_count} / {total_pairs}")
    st.write(f"- Intersection ratio: {intersection_ratio:.1f}%")
    st.write(f"- Avg neighbors/sphere: {2*intersection_count/len(spheres):.2f}")

# 3D Visualization
st.subheader("🎨 3D Lattice Visualization")

fig = go.Figure()

# Add spheres (as points for simplicity)
fig.add_trace(go.Scatter3d(
    x=positions[:, 0],
    y=positions[:, 1],
    z=positions[:, 2],
    mode='markers+text',
    marker=dict(
        size=5,
        color=kmax,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="k_max")
    ),
    text=[f"S{i}" for i in range(len(positions))],
    textposition="top center",
    hovertemplate="<b>Sphere %{text}</b><br>Pos: (%{x:.2f}, %{y:.2f}, %{z:.2f})"
))

fig.update_layout(
    title=f"{lattice_name} Lattice ({system_size}) - a={a:.2f}Å, r={r:.2f}Å",
    scene=dict(
        xaxis_title="X (Ångströms)",
        yaxis_title="Y (Ångströms)",
        zaxis_title="Z (Ångströms)",
        aspectmode='data'
    ),
    width=800,
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Info section
with st.expander("ℹ️ About This Tool"):
    st.markdown("""
    This tool calculates the **maximum sphere intersection order (k_max)** for crystal lattices using the Radical Center method.
    
    **How it works:**
    - For each triplet of spheres, finds their radical center (point equidistant from all three)
    - Checks how many spheres pass through this point
    - Reports the maximum intersection order found
    
    **Lattice Types:**
    - **FCC**: Face-Centered Cubic (coordination number 12)
    - **BCC**: Body-Centered Cubic (coordination number 8)
    - **SC**: Simple Cubic (coordination number 6)
    
    **Use Cases:**
    - Analyze coordination geometry in crystal structures
    - Find optimal sphere overlap parameters
    - Map parameter space to identify phase transitions
    - Validate crystallographic models
    
    **Algorithm:** O(N^4) Radical Center Method
    - Efficient for N ≤ 100 spheres
    - Numerically stable (EPSILON = 1e-10)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
<p>Radical Center k_max Calculator | Built with Streamlit</p>
<p><a href="https://github.com">GitHub</a> • <a href="#">Documentation</a></p>
</div>
""", unsafe_allow_html=True)
