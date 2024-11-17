from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from image_matching import process_image, cache_dataset_features, load_dataset_features

UPLOAD_FOLDER = 'uploads'
DATASET_FOLDER = 'dataset'
CACHE_FILE = 'dataset_features.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cache the dataset features if not already cached
if not os.path.exists(CACHE_FILE):
    cache_dataset_features(DATASET_FOLDER, CACHE_FILE)

# Load the cached dataset features
dataset_features = load_dataset_features(CACHE_FILE)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        results = process_image(file_path, dataset_features)
        return render_template('result.html', results=results, image=filename)
    else:
        return redirect(url_for('home'))

@app.route('/templates/<path:filename>')
def serve_template_file(filename):
    return send_from_directory('templates', filename)

if __name__ == "__main__":
    app.run(debug=True)
