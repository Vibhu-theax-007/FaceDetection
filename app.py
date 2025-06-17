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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists

# ---- INITIALIZE FACE ANALYSIS ----
face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=(640, 640))

# ---- LOAD IMAGE UTILS ----
def load_image(path):
    try:
        pil_image = Image.open(path)
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"❌ Error loading image '{path}': {e}")
        return None

def get_embedding(image):
    # Extract face embeddings using insightface
    faces = face_app.get(image)
    if not faces:
        print("❌ No face detected in the image.")
        return None
    # Normalize the embedding vector
    return faces[0].embedding / np.linalg.norm(faces[0].embedding)

# ---- LOAD REFERENCE IMAGES ONCE ----
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
        emb = get_embedding(img)
        if emb is None:
            print(f"❌ No face found in: {filename}")
            continue
        reference_embeddings[filename] = emb
        print(f"✅ Loaded and embedded: {filename}")

print("✅ Reference images ready:", list(reference_embeddings.keys()))

# ---- ROUTES ----
@app.route('/')
def index():
    return render_template('capture.html')  # Directly open camera page

@app.route('/verify', methods=['POST'])
def verify():
    try:
        # Get the image from the frontend
        data = request.form['image']
        encoded = data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        live_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        print("✅ Live image received and decoded.")

        # Get embedding for the live image
        live_emb = get_embedding(live_image)
        if live_emb is None:
            return "❌ No face detected in the captured image."

        print(f"✅ Live image embedding: {live_emb}")

        # Compare live embedding with reference embeddings
        max_score = 0
        matched_person = "Unknown"
        threshold = 0.6  # Adjusted threshold

        for name, ref_emb in reference_embeddings.items():
            similarity = np.dot(ref_emb, live_emb)
            print(f"Comparing {name}: Similarity: {similarity:.4f}")
            if similarity > max_score:
                max_score = similarity
                matched_person = name

        if max_score >= threshold:
            return f"✅ Match found: {matched_person} (Similarity: {max_score:.4f})"
        else:
            return f"❌ No match found. Closest: {matched_person} (Score: {max_score:.4f})"

    except Exception as e:
        print(f"❌ Error during verification: {str(e)}")
        return f"❌ Error during verification: {str(e)}"

# ---- RUN APP ----
if __name__ == '__main__':
    print("🚀 Starting Flask server...")
    app.run(debug=True)
