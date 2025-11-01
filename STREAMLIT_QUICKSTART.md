# How to Run the Streamlit UI

## Local (Laptop/Desktop)

### 1. Install dependencies
```bash
pip install streamlit plotly numpy
```

### 2. Run the app
```bash
streamlit run streamlit_app.py
```

App opens automatically at `http://localhost:8501`

---

## Cloud (Recommended - No Installation)

### Option A: Streamlit Cloud (FREE)
1. Sign up at https://streamlit.io/cloud
2. Connect GitHub repo with `streamlit_app.py`
3. Click "Deploy"
4. Runs forever, always accessible

### Option B: Google Colab
```python
!pip install streamlit plotly -q
!streamlit run streamlit_app.py &
```
Gets ngrok URL to access from browser

### Option C: Replit (FREE)
1. Upload files to https://replit.com
2. Create `requirements.txt`:
   ```
   streamlit
   plotly
   numpy
   ```
3. Run: `streamlit run streamlit_app.py`

### Option D: Heroku/Railway (FREE tier)
Deploy in 2 minutes, runs 24/7

---

## Features

✓ **Slider controls** for lattice parameter (a) and radius (r)
✓ **3 lattice types** - FCC, BCC, SC
✓ **4 system sizes** - 2×2×2 to 4×4×4
✓ **Real-time calculation** - instant results
✓ **3D visualization** - interactive lattice view
✓ **Geometric analysis** - nearest neighbor distances
✓ **Intersection statistics** - pair analysis

---

## Usage

1. Select **Lattice Type** (FCC/BCC/SC)
2. Choose **System Size** (2×2×2, 3×3×3, 4×4×4)
3. Adjust **Lattice Parameter (a)** slider
4. Adjust **Radius (r)** slider
5. **See k_max result instantly**
6. Explore 3D visualization
7. View geometric analysis

---

## Deployment (30 seconds)

**Easiest:** Streamlit Cloud
```bash
git push  # Streamlit auto-deploys on push
```

Result: Your own URL like `https://yourusername-appname.streamlit.app`

---

## Troubleshooting

**Module not found?**
- Copy `radical_center_enhanced.py` to same folder as `streamlit_app.py`

**Port 8501 already in use?**
```bash
streamlit run streamlit_app.py --server.port 8502
```

**Slow on large systems?**
- Use 2×2×2 for instant results
- 3×3×3 takes 1-2 seconds
- 4×4×4 takes 5-10 seconds

---

**Need help?** All files in `/mnt/user-data/outputs/`
