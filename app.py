from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import base64
from PIL import Image
from insightface.app import FaceAnalysis

app = Flask(__name__)

# ---- CONFIGURATION ----
UPLOAD_FOLDER = 'reference'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- INITIALIZE FACE ANALYSIS ----
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ---- IMAGE UTILITIES ----
def load_image(path):
    try:
        pil_image = Image.open(path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Error loading image '{path}': {e}")
        return None

def get_embeddings(image):
    faces = face_app.get(image)
    if not faces:
        return []
    return [face.embedding / np.linalg.norm(face.embedding) for face in faces]

# ---- LOAD REFERENCE IMAGES ----
reference_embeddings = {}
print("📦 Loading reference images...")
for filename in os.listdir(UPLOAD_FOLDER):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"➡️ Processing: {path}")
        img = load_image(path)
        if img is None:
            print(f"❌ Failed to load: {filename}")
            continue
        embs = get_embeddings(img)
        if not embs:
            print(f"❌ No face found in: {filename}")
            continue
        reference_embeddings[filename] = embs[0]  # Use the first face
        print(f"✅ Loaded and embedded: {filename}")
print("✅ Reference images ready:", list(reference_embeddings.keys()))

# ---- ROUTES ----
@app.route('/')
def index():
    return render_template('capture.html')

@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.form['image']
        encoded = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        live_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        live_embs = get_embeddings(live_image)
        if not live_embs:
            return "❌ No face detected in the captured image."

        results = []
        threshold = 0.5

        for i, live_emb in enumerate(live_embs):
            best_score = 0
            matched_name = "Unknown"
            for name, ref_emb in reference_embeddings.items():
                score = np.dot(ref_emb, live_emb)
                if score > best_score:
                    best_score = score
                    matched_name = name
            if best_score >= threshold:
                results.append(f"✅ Face {i+1}: {matched_name} (Similarity: {best_score:.4f})")
            else:
                results.append(f"❌ Face {i+1}: Unknown (Closest: {matched_name}, Score: {best_score:.4f})")

        return "<br>".join(results)

    except Exception as e:
        return f"❌ Error during verification: {str(e)}"

# ---- RUN APP ----
if __name__ == '__main__':
    print("🚀 Starting Face Auth App...")
    app.run(debug=True)
