# app.py
# Simple Flask web app for crowd anomaly prediction and visualization

from flask import Flask, request, render_template_string, send_file
from PIL import Image
import pickle
import os
import io
import math
import base64
from PIL import ImageDraw
import numpy as np
from flow_estimation import estimate_flow, dominant_direction
from bottleneck_detection import detect_bottlenecks
from movement_instruction import generate_instructions, generate_instruction_details
from venue_config import EXITS

app = Flask(__name__)

# Load model weights and PCA
with open("autoencoder_weights.pkl", "rb") as f:
    weights = pickle.load(f)
with open("pca_results.pkl", "rb") as f:
    pca = pickle.load(f)

# Autoencoder class (same as before, but loads weights)
class Autoencoder:
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
        self.input_size = len(W2)
        self.hidden_size = len(W1)
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    def encode(self, x):
        h = [0.0 for _ in range(self.hidden_size)]
        for i in range(self.hidden_size):
            h[i] = self.sigmoid(sum(self.W1[i][j] * x[j] for j in range(self.input_size)) + self.b1[i])
        return h
    def decode(self, h):
        y = [0.0 for _ in range(self.input_size)]
        for i in range(self.input_size):
            y[i] = self.sigmoid(sum(self.W2[i][j] * h[j] for j in range(self.hidden_size)) + self.b2[i])
        return y
    def reconstruct(self, x):
        h = self.encode(x)
        y = self.decode(h)
        return y

def preprocess_image(img, size=(64,64)):
    img = img.convert('L').resize(size)
    return [pixel/255.0 for pixel in img.getdata()]

def anomaly_map(input_vec, recon_vec, size=(64,64)):
    diff = [abs(a-b) for a,b in zip(input_vec, recon_vec)]
    max_diff = max(diff)
    img_data = [int(255 * (d / max_diff)) if max_diff > 0 else 0 for d in diff]
    img = Image.new('L', size)
    img.putdata(img_data)
    return img

def pca_reconstruction(vec, pc1, means):
    proj = sum((vec[j] - means[j]) * pc1[j] for j in range(len(pc1)))
    recon = [means[j] + proj * pc1[j] for j in range(len(pc1))]
    return recon

def reconstruction_error(vec, recon):
    return math.sqrt(sum((vec[j] - recon[j])**2 for j in range(len(vec))))

@app.route('/', methods=['GET', 'POST'])
def index():
    def draw_flow_arrow(img, flow):
        # Overlay flow arrows on the frame for better visualization
        base = img.convert('RGB').resize((128,128))
        draw = ImageDraw.Draw(base)
        h, w = flow.shape
        flow_np = np.array(flow)
        
        # Calculate overall flow direction
        y_indices, x_indices = np.mgrid[0:h, 0:w]
        total = np.sum(flow_np)
        if total > 1e-6:
            x_flow = np.sum(flow_np * x_indices) / total - w/2
            y_flow = np.sum(flow_np * y_indices) / total - h/2
            
            # Scale for visualization
            scale = 20
            center_x, center_y = 64, 64  # Center of 128x128 image
            end_x = int(center_x + scale * x_flow)
            end_y = int(center_y + scale * y_flow)
            
            # Draw main flow arrow
            if abs(x_flow) > 0.1 or abs(y_flow) > 0.1:
                draw.line([center_x, center_y, end_x, end_y], fill=(255,0,0), width=3)
                # Draw arrowhead
                arrow_size = 5
                if abs(x_flow) > abs(y_flow):
                    if x_flow > 0:  # Right
                        draw.polygon([(end_x, end_y), (end_x-arrow_size, end_y-arrow_size), (end_x-arrow_size, end_y+arrow_size)], fill=(255,0,0))
                    else:  # Left
                        draw.polygon([(end_x, end_y), (end_x+arrow_size, end_y-arrow_size), (end_x+arrow_size, end_y+arrow_size)], fill=(255,0,0))
                else:
                    if y_flow > 0:  # Down
                        draw.polygon([(end_x, end_y), (end_x-arrow_size, end_y-arrow_size), (end_x+arrow_size, end_y-arrow_size)], fill=(255,0,0))
                    else:  # Up
                        draw.polygon([(end_x, end_y), (end_x-arrow_size, end_y+arrow_size), (end_x+arrow_size, end_y+arrow_size)], fill=(255,0,0))
        
        return base
    result_html = ''
    error_html = ''
    instructions_html = ''
    bottleneck_html = ''
    summary_html = ''
    if request.method == 'POST':
        try:
            files = request.files.getlist('image')
            alpha = float(request.form.get('alpha', '0.75'))
            dens_th = float(request.form.get('density_threshold', '0.55'))
            move_th = float(request.form.get('movement_threshold', '0.04'))
            images = []
            orig_imgs = []
            for file in files:
                img = Image.open(file.stream)
                orig_imgs.append(img.copy())
                images.append(np.array(img.convert('L').resize((64,64)), dtype=np.float32) / 255.0)
            flows = estimate_flow(images)
            bottlenecks = detect_bottlenecks(images, flows, density_threshold=dens_th, movement_threshold=move_th)
            details = generate_instruction_details(images, flows, bottlenecks, alpha=alpha)
            # Build cards per frame
            cards = ''
            for d in details:
                conf_badge = f"<span class='badge bg-secondary ms-2'>Confidence: {int(d.get('confidence',0)*100)}%</span>"
                badge = ('<span class="badge bg-warning text-dark">Bottleneck</span>' if d['bottleneck'] else '<span class="badge bg-info">Info</span>') + conf_badge
                cards += (
                    '<div class="col">\n'
                    '  <div class="card mb-3" style="width: 16rem;">\n'
                    f'    <div class="card-header d-flex justify-content-between align-items-center">Frame {d["frame"]} {badge}</div>\n'
                    '    <div class="card-body">\n'
                    f'      <div><strong>Flow:</strong> {d["flow_text"]}</div>\n'
                    f'      <div><strong>Zone:</strong> {d["zone"]}</div>\n'
                    f'      <div><strong>Exit:</strong> {d["exit"]}</div>\n'
                    f'      <div class="mt-2"><strong>Action:</strong> {d["recommendation"]}</div>\n'
                    '    </div>\n'
                    '  </div>\n'
                    '</div>'
                )
            instructions_html = '<div class="row row-cols-auto">' + cards + '</div>'
            # Summary stats
            try:
                num_b = sum(1 for d in details if d.get('bottleneck'))
                confs = [float(d.get('confidence',0)) for d in details if 'confidence' in d]
                avg_c = int(100 * (sum(confs)/len(confs))) if confs else 0
            except Exception:
                num_b, avg_c = 0, 0
            summary_html = (
                '<div class="alert alert-light border d-flex justify-content-between align-items-center">'
                f'<div><strong>Summary:</strong> Bottlenecks detected: <span class="badge bg-warning text-dark">{num_b}</span></div>'
                f'<div>Average confidence: <span class="badge bg-secondary">{avg_c}%</span></div>'
                '</div>'
            )
            # Show original images with flow arrows
            img_html = ''
            for i, img in enumerate(orig_imgs):
                buf = io.BytesIO()
                if i > 0 and i-1 < len(flows):
                    img_arrow = draw_flow_arrow(img.copy(), flows[i-1])
                else:
                    img_arrow = img.resize((128,128))
                img_arrow.save(buf, format='PNG')
                img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                img_html += f'<div class="col"><label>Frame {i}</label><br><img src="data:image/png;base64,{img_b64}" class="img-thumbnail" width=128></div>'
            result_html = (
                '<h5 class="mt-3">Flow Visualization</h5><div class="row overflow-auto" style="max-height:60vh">' + img_html + '</div>'
            )
        except Exception as e:
            error_html = f'<div class="alert alert-danger">Error: {str(e)}</div>'
    return render_template_string(
        '''<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
          body { background: #f6f8fb; }
          .navbar { background: linear-gradient(90deg, #0d6efd, #6610f2); }
          .navbar .navbar-brand { color: #fff; font-weight: 600; }
          .card { box-shadow: 0 0.25rem 0.75rem rgba(0,0,0,0.05); border: none; }
          .card-header { background: #f1f5ff; font-weight: 600; }
        </style>
        <nav class="navbar navbar-expand-lg mb-3">
          <div class="container">
            <span class="navbar-brand">Crowd Guidance Dashboard</span>
          </div>
        </nav>
        <div class="container">
        <div class="alert alert-secondary py-2">
          <strong>Legend:</strong>
          <span class="badge bg-warning text-dark ms-2">Bottleneck</span>
          <span class="badge bg-info ms-2">Info</span>
          <span class="badge bg-secondary ms-2">Confidence</span>
          <span class="ms-3"><small>Exits: ''' + ", ".join(EXITS.keys()) + '''</small></span>
        </div>
        <form method="post" enctype="multipart/form-data" class="mb-3">
          <input type="file" name="image" accept="image/*" multiple required class="form-control mb-2">
          <div class="row g-2 mb-2">
            
          </div>
          <input type="submit" value="Analyze Crowd Flow" class="btn btn-primary">
        </form>
        ''' + summary_html + '''
        <div class="row">
          <div class="col-lg-6">
            <h5>Guidance</h5>
            ''' + instructions_html + '''
          </div>
          <div class="col-lg-6">
            ''' + result_html + '''
          </div>
        </div>
        </div>'''
    )

if __name__ == '__main__':
    app.run(debug=True)
