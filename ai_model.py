import os
import io
import numpy as np
import spectral.io.envi as envi
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from PIL import Image
from scipy.interpolate import interp1d
MODEL_PATH = 'ai_model.h5'
REFS_FILE = 'references_master.npy'   

CUBE_HDR = r'output_cube3/cube_test.hdr'
CUBE_RAW = r'output_cube3/cube_test.raw'

OUTPUT_IMAGE = 'AI_model_output.bmp'
TARGET_WIDTH = 2592
TARGET_HEIGHT = 1944

CNN_THRESHOLD = 0.95
SAM_THRESHOLD = 0.40
MODEL_WAVELENGTHS = np.arange(480, 891, 10) 

CLASSES = ['Ilmenite', 'Olivine', 'Pyroxene', 'Unknown']
COLORS = ['#D32F2F', "#29DB32", "#A14FC4", '#212121']


def resample_cube_to_model(cube_obj, target_wavs):
    
    try:
        source_wavs = np.array([float(w) for w in cube_obj.metadata['wavelength']])
    except KeyError:
        pass

    if source_wavs.max() < 100: source_wavs *= 1000
    
    data = cube_obj.load()
    rows, cols, bands = data.shape
    X_raw = data.reshape(-1, bands)

    resampler = interp1d(source_wavs, X_raw, kind='linear', axis=1, bounds_error=False, fill_value="extrapolate")
    X_resampled = resampler(target_wavs)
    return X_resampled, (rows, cols)

def spectral_angle_mapper(v1, v2):
    v1 = v1.flatten(); v2 = v2.flatten()
    nom = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.arccos(np.clip(nom / (denom + 1e-8), -1.0, 1.0))

def generate_dashboard():
    if not os.path.exists(REFS_FILE):
        pass
        
    refs = np.load(REFS_FILE, allow_pickle=True).item()
    

    cube_obj = envi.open(CUBE_HDR, CUBE_RAW)
    X_resampled, shape = resample_cube_to_model(cube_obj, MODEL_WAVELENGTHS)
    rows, cols = shape
    
    X_min = X_resampled.min(axis=1, keepdims=True)
    X_max = X_resampled.max(axis=1, keepdims=True)
    X_norm = (X_resampled - X_min) / (X_max - X_min + 1e-8)
    X_input = X_norm.reshape(-1, len(MODEL_WAVELENGTHS), 1)

    model = load_model(MODEL_PATH)
    preds_prob = model.predict(X_input, verbose=1)
    
    final_map_flat = []
    unknown_idx = 3
    
    for i in range(len(preds_prob)):
        prob = np.max(preds_prob[i])
        idx = np.argmax(preds_prob[i])
        name = CLASSES[idx].lower() 
        
        
        if prob < CNN_THRESHOLD:
            final_map_flat.append(unknown_idx)

        elif name in refs and spectral_angle_mapper(X_norm[i], refs[name]) > SAM_THRESHOLD:
            final_map_flat.append(unknown_idx)
        else:
            final_map_flat.append(idx)

    final_map = np.array(final_map_flat).reshape(rows, cols)

    
    my_dpi = 100
    fig = plt.figure(figsize=(TARGET_WIDTH/my_dpi, TARGET_HEIGHT/my_dpi), dpi=my_dpi)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.6, 1], wspace=0.05, hspace=0.15)

    
    ax1 = plt.subplot(gs[0, 0])
    bands = len(MODEL_WAVELENGTHS)
    try:
        r = int(bands*0.85); g = int(bands*0.5); b = int(bands*0.1)
        rgb = np.dstack((X_norm.reshape(rows,cols,bands)[:,:,r], 
                         X_norm.reshape(rows,cols,bands)[:,:,g], 
                         X_norm.reshape(rows,cols,bands)[:,:,b]))
        ax1.imshow(rgb)
    except: ax1.imshow(np.zeros((rows, cols)))
    ax1.set_title("1. HSI Data (False RGB)", fontsize=28, fontweight='bold', pad=15)
    ax1.axis('off')

    # 2. Classification Map
    ax2 = plt.subplot(gs[0, 1])
    cmap = ListedColormap(COLORS)
    ax2.imshow(final_map, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
    ax2.set_title("2. Mineral Map", fontsize=28, fontweight='bold', pad=15)
    ax2.axis('off')

    # 3. Pie Chart
    ax3 = plt.subplot(gs[1, 0])
    u, c = np.unique(final_map, return_counts=True)
    counts = dict(zip(u, c))
    p_d = []; p_l = []; p_c = []
    for i in range(4):
        if i in counts and counts[i] > 0:
            p_d.append(counts[i]); p_l.append(CLASSES[i]); p_c.append(COLORS[i])
            
    if p_d:
        wedges, texts, autotexts = ax3.pie(p_d, labels=p_l, colors=p_c, autopct='%1.1f%%', 
                                           textprops={'fontsize': 22, 'weight': 'bold'})
        for at in autotexts: at.set_color('white'); at.set_fontsize(20)
    ax3.set_title("3. Composition %", fontsize=28, fontweight='bold', pad=15)

    # 4. Legend
    ax4 = plt.subplot(gs[1, 1]); ax4.axis('off')
    patches = [mpatches.Patch(color=COLORS[i], label=CLASSES[i]) for i in range(4)]
    leg = ax4.legend(handles=patches, loc='center', fontsize=32, title="Key", title_fontsize=36)

    plt.suptitle("Universal AI Mineralogy Report", fontsize=48, fontweight='bold', y=0.96)

    # Save
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=my_dpi, facecolor='white')
    buf.seek(0)
    img = Image.open(buf).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    img.save(OUTPUT_IMAGE, "BMP")
    

if __name__ == "__main__":
    generate_dashboard()