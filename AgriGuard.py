import os
from flask import Flask, request, render_template_string, send_file
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import io
import base64
from datetime import datetime
from fpdf import FPDF
import re

# === CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "agriguard_model.h5")

app = Flask(__name__)

# === LOAD MODEL ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
full_model = tf.keras.models.load_model(MODEL_PATH)
base_model = full_model.layers[0]
input_tensor = tf.keras.Input(shape=(224,224,3))
features = base_model(input_tensor)
x = full_model.layers[1](features)
x = full_model.layers[2](x)
predictions = full_model.layers[3](x)
grad_model = tf.keras.Model(inputs=input_tensor, outputs=[features, predictions])

# === CLASS INFO ===
CLASS_NAMES = [
    'Corn___Cercospora_Leaf_Spot','Corn___Common_Rust','Corn___Healthy','Corn___Northern_Leaf_Blight',
    'Potato___Early_Blight','Potato___Healthy','Potato___Late_Blight',
    'Tomato___Bacterial_Spot','Tomato___Early_Blight','Tomato___Healthy','Tomato___Late_Blight',
    'Tomato___Septoria_Leaf_Spot','Tomato___Yellow_Leaf_Curl_Virus'
]

DESCRIPTIONS = {k:v for k,v in zip(CLASS_NAMES,[
    "Grayish spots with dark borders on leaves, often merging into large dead areas.",
    "Small, circular, reddish-brown pustules on upper leaf surfaces.",
    "Vibrant green leaves with no discoloration or lesions.",
    "Large, cigar-shaped gray-green lesions that turn tan.",
    "Dark brown spots with concentric rings ('target spot') on older leaves.",
    "Lush green foliage with no blemishes.",
    "Water-soaked, gray-green lesions; white mold on undersides in wet weather.",
    "Small, dark, water-soaked spots; may have yellow halos.",
    "Dark spots with concentric rings; yellowing around lesions.",
    "Uniform green leaves, sturdy stems.",
    "Large, irregular, water-soaked lesions; white fungal growth under leaves.",
    "Small, circular spots with gray centers and dark margins.",
    "Severe upward leaf curling, yellowing, stunted growth."
])}

REMEDIES = {k:v for k,v in zip(CLASS_NAMES,[
    "Use certified disease-free seeds. Apply fungicides like mancozeb early.",
    "Plant resistant hybrids. Apply foliar fungicides if rust >5% of leaf area.",
    "Continue good practices: proper spacing and balanced fertilization.",
    "Remove infected debris. Rotate crops. Apply fungicide at tasseling.",
    "Mulch to reduce soil splash. Apply copper-based fungicides preventively.",
    "Maintain consistent watering and soil health.",
    "Use certified seed tubers. Destroy cull piles. Apply fungicides before rain.",
    "Avoid overhead watering. Use copper sprays. Remove infected plants.",
    "Prune lower leaves for airflow. Apply chlorothalonil or biofungicides.",
    "Great job! Keep watering at the base and rotating crops.",
    "Destroy infected plants immediately. Do not compost. Use resistant varieties.",
    "Remove lower leaves. Avoid wetting foliage. Apply fungicide early.",
    "Control whiteflies with sticky traps or insecticidal soap."
])}

DISEASE_COLORS = {
    'Corn___Cercospora_Leaf_Spot': (100,200,255),'Corn___Common_Rust': (0,165,255),'Corn___Healthy': (0,255,0),
    'Corn___Northern_Leaf_Blight': (226,43,138),'Potato___Early_Blight': (0,165,255),'Potato___Healthy': (0,255,0),
    'Potato___Late_Blight': (0,0,255),'Tomato___Bacterial_Spot': (0,0,255),'Tomato___Early_Blight': (42,42,165),
    'Tomato___Healthy': (0,255,0),'Tomato___Late_Blight': (0,0,255),'Tomato___Septoria_Leaf_Spot': (0,165,255),
    'Tomato___Yellow_Leaf_Curl_Virus': (0,255,255)
}

# === GRAD-CAM & PREDICTION ===
def get_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None: pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap,0)/tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap(image, heatmap):
    heatmap = cv2.resize(heatmap,(image.shape[1],image.shape[0]))
    heatmap_uint8 = (heatmap*255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image.astype(np.uint8),0.7,heatmap_color,0.3,0)
    return blended, heatmap_uint8

def get_hotspots_and_severity(binary_mask):
    contours,_ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hotspots=[]
    total_area=0
    for i,c in enumerate(contours[:3],1):
        M=cv2.moments(c)
        if M['m00']==0: continue
        cx=int(M['m10']/M['m00']); cy=int(M['m01']/M['m00'])
        hotspots.append(f"📍 Lesion {i} at ({cx},{cy})")
        total_area+=cv2.contourArea(c)
    severity=round(min(100,total_area/5000*100),1) # crude severity estimate
    return "<br>".join(hotspots) if hotspots else "No significant disease regions detected.", severity

def predict_with_heatmap(img):
    original_img=np.array(img)
    img_resized=img.resize((224,224))
    img_array=np.array(img_resized)/255.0
    img_batch=np.expand_dims(img_array,axis=0)
    preds=full_model.predict(img_batch,verbose=0)[0]
    idx=np.argmax(preds)
    label=CLASS_NAMES[idx]
    confidence=float(preds[idx])
    heatmap=get_gradcam_heatmap(img_batch,grad_model,pred_index=idx)
    blended_img, heatmap_uint8 = apply_heatmap(original_img,heatmap)
    _, binary_mask = cv2.threshold(heatmap_uint8,150,255,cv2.THRESH_BINARY)
    hotspots,severity = get_hotspots_and_severity(binary_mask)

    # AI suggestion based on severity
    if severity>30:
        suggestion="Apply treatment immediately and monitor crop closely."
    elif severity>10:
        suggestion="Apply preventive measures and monitor disease progression."
    else:
        suggestion="Minor symptoms detected; monitor regularly."

    return label, confidence, blended_img, hotspots, severity, suggestion


# === HTML TEMPLATE ===
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgriGuard – AI Crop Doctor</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
body{background:#f0fdf4;font-family:Inter,sans-serif;scroll-behavior:smooth;padding:1rem;}
.drag-area{border:3px dashed #84cc16;border-radius:1rem;padding:2rem;text-align:center;transition:0.3s;cursor:pointer;}
.drag-area.dragover{background:rgba(132,204,22,0.1);}
.card-agri{background:white;border-radius:1rem;padding:1.5rem;box-shadow:0 6px 18px rgba(0,0,0,0.1);}
.btn-agri{background:linear-gradient(135deg,#84cc16,#65a30d);color:white;padding:0.5rem 1.5rem;border-radius:0.75rem;font-weight:600;transition:0.2s;}
.btn-agri:hover{transform:translateY(-2px);box-shadow:0 6px 12px rgba(132,204,22,0.3);}
.legend-toggle{position:absolute;top:10px;right:10px;background:rgba(255,255,255,0.8);padding:0.3rem 0.5rem;border-radius:0.5rem;cursor:pointer;font-size:0.8rem;font-weight:600;}
.legend{position:absolute;top:40px;right:10px;background:rgba(255,255,255,0.9);padding:0.5rem;border-radius:0.75rem;max-height:300px;overflow-y:auto;display:none;}
.legend div{display:flex;align-items:center;margin-bottom:0.3rem;}
.legend div span{width:20px;height:15px;display:inline-block;margin-right:0.5rem;}
.lesion-tooltip{width:12px;height:12px;background:rgba(255,0,0,0.4);border-radius:50%;display:inline-block;position:absolute;transform:translate(-50%,-50%);}
.lesion-tooltip:hover::after{content:attr(data-label);position:absolute;top:-1.5rem;left:50%;transform:translateX(-50%);background:rgba(0,0,0,0.7);color:white;padding:2px 6px;font-size:0.75rem;border-radius:4px;white-space:nowrap;}
</style>
</head>
<body>
<div class="max-w-6xl mx-auto relative">
<header class="text-center mb-8">
<h1 class="text-5xl font-bold text-green-800 mb-2">🌱 AgriGuard</h1>
<p class="text-gray-700 text-lg">AI-powered crop disease detection with <strong>precise lesion localization</strong> and <strong>actionable insights</strong></p>
</header>

<div class="card-agri mb-6">
<div id="drag-area" class="drag-area mb-4">
<h3 class="text-2xl font-semibold mb-2">📸 Upload or Drag a Leaf Photo</h3>
<p class="text-gray-500 mb-3">Drag your image here or click to select</p>
<form id="upload-form" method="post" enctype="multipart/form-data">
<input type="file" name="file" accept="image/*" id="file-input" class="hidden"/>
<button type="button" class="btn-agri" onclick="document.getElementById('file-input').click()">Select File</button>
</form>
</div>
</div>

{% if result %}
<div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-10 relative">
  <!-- Original Image -->
  <div class="card-agri p-4">
    <h4 class="font-semibold mb-2">📸 Original Leaf</h4>
    <img src="data:image/jpeg;base64,{{ result.original_b64 }}" class="rounded-lg w-full object-cover h-64">
  </div>

  <!-- Heatmap / AI Lesion Detection -->
  <div class="card-agri p-4 relative">
    <h4 class="font-semibold mb-2">🔍 AI Lesion Detection</h4>
    <div class="relative">
      <img src="data:image/jpeg;base64,{{ result.heatmap_b64 }}" class="rounded-lg w-full object-cover h-64">
      <div class="legend-toggle" onclick="toggleLegend()">Legend</div>
      <div class="legend" id="legend">
        {% for disease,color in disease_colors.items() %}
        <div><span style="background:rgb({{ color[2] }},{{ color[1] }},{{ color[0] }});"></span>{{ disease.replace('___',' – ').replace('_',' ') }}</div>
        {% endfor %}
      </div>
      <div class="absolute top-0 left-0 w-full h-full">
        {% for spot in result.hotspot_positions %}
          <span class="lesion-tooltip" style="left:{{ spot.x/500*100 }}%; top:{{ spot.y/500*100 }}%;" data-label="{{ spot.label.replace('___',' – ').replace('_',' ') }}"></span>
        {% endfor %}
      </div>
    </div>
    <p class="text-xs text-gray-500 mt-1">🎨 Color-coded lesions + legend overlay</p>
  </div>

  <!-- Diagnosis & Actions -->
  <div class="card-agri p-4">
    <h3 class="font-bold mb-2">🧠 Diagnosis</h3>
    <p><strong>Disease:</strong> {{ result.label }}</p>
    <p><strong>Confidence:</strong> {{ "%.1f"|format(result.confidence*100) }}%</p>
    <p><strong>Visual Signs:</strong> {{ result.description }}</p>
    <p><strong>Recommended Action:</strong> {{ result.remedy }}</p>

    <p><strong>Severity Score:</strong> 
      <span class="{% if result.severity > 30 %}text-red-600{% elif result.severity > 10 %}text-yellow-600{% else %}text-green-600{% endif %}">
        {{ result.severity }}%
      </span>
    </p>

    <p><strong>AI Suggestion:</strong> <span class="text-blue-700">{{ result.suggestion }}</span></p>

    <p><strong>Detected Lesions:</strong><br>{{ result.hotspots | safe }}</p>

    <!-- Download PDF -->
    <button type="button" onclick="downloadReport()" class="btn-agri mt-2">📥 Download Report</button>
    <form id="download-form" method="POST" action="/download_pdf" style="display:none;">
        <input type="hidden" name="result_json" id="result-json">
    </form>

    <a href="/" class="btn-agri mt-2 ml-2 inline-block">Analyze Another</a>
  </div>
</div>
{% endif %}

</div>

<script>
// Toggle Legend
function toggleLegend(){
    const legend = document.getElementById('legend');
    legend.style.display = legend.style.display==='block'?'none':'block';
}

// PDF Download (only if result exists)
{% if result %}
function downloadReport(){
    const result = {{ result|tojson }};
    document.getElementById('result-json').value = JSON.stringify(result);
    document.getElementById('download-form').submit();
}
{% endif %}

// Drag & Drop
const dragArea=document.getElementById('drag-area');
const fileInput=document.getElementById('file-input');
dragArea.addEventListener('dragover', e => { e.preventDefault(); dragArea.classList.add('dragover'); });
dragArea.addEventListener('dragleave', e => { dragArea.classList.remove('dragover'); });
dragArea.addEventListener('drop', e => {
    e.preventDefault();
    dragArea.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if(file){
        const dt = new DataTransfer(); 
        dt.items.add(file);
        fileInput.files = dt.files; 
        document.getElementById('upload-form').submit();
    }
});
fileInput.addEventListener('change', () => document.getElementById('upload-form').submit());
</script>
</body>
</html>
'''

# === PDF GENERATION (Unicode-safe) ===
def clean_text(text):
    """Remove or replace characters not supported by core PDF fonts."""
    # Replace en-dash with normal dash
    text = text.replace('–', '-')
    # Remove emojis and any other non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def generate_pdf_report(result):
    pdf = FPDF()
    pdf.add_page()
    
    # === TITLE ===
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, clean_text('AgriGuard Crop Diagnosis Report'), ln=1, align='C')
    
    # Date
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 6, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1)
    pdf.ln(5)
    
    # Original Image
    img_buf = io.BytesIO(base64.b64decode(result['original_b64']))
    img = Image.open(img_buf)
    img.save('temp_orig.jpg')
    pdf.image('temp_orig.jpg', w=pdf.w-40, x=20)
    os.remove('temp_orig.jpg')
    pdf.ln(5)
    
    # Diagnosis section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 8, clean_text('Diagnosis'), ln=1)
    
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(0, 6, clean_text(
        f"Disease: {result['label']}\n"
        f"Confidence: {result['confidence']*100:.1f}%\n"
        f"Severity: {result['severity']}%\n"
        f"Visual Signs: {result['description']}\n"
        f"Recommended Action: {result['remedy']}\n"
        f"AI Suggestion: {result['suggestion']}\n"
        f"Detected Lesions: {result['hotspots']}"
    ))
    
    # Heatmap Image
    img_buf = io.BytesIO(base64.b64decode(result['heatmap_b64']))
    img = Image.open(img_buf)
    img.save('temp_heat.jpg')
    pdf.image('temp_heat.jpg', w=pdf.w-40, x=20)
    os.remove('temp_heat.jpg')
    
    # Return as BytesIO
    buf = io.BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf


# === FLASK ROUTE ===
@app.route('/',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        file=request.files.get('file')
        if not file: return "No file uploaded",400
        try:
            img=Image.open(file.stream).convert('RGB')
            label,conf,heatmap_img,hotspots,severity,suggestion=predict_with_heatmap(img)

            buf_orig=io.BytesIO(); img.save(buf_orig,format='JPEG'); orig_b64=base64.b64encode(buf_orig.getvalue()).decode()
            buf_heat=io.BytesIO(); heatmap_pil=Image.fromarray(heatmap_img.astype('uint8'),'RGB'); heatmap_pil.save(buf_heat,format='JPEG')
            heatmap_b64=base64.b64encode(buf_heat.getvalue()).decode()

            display_label=label.replace('___',' – ').replace('_',' ')

            result={'label':display_label,'confidence':conf,'description':DESCRIPTIONS.get(label,''),'remedy':REMEDIES.get(label,''),
                    'original_b64':orig_b64,'heatmap_b64':heatmap_b64,'hotspots':hotspots,'severity':severity,'suggestion':suggestion}

            return render_template_string(HTML_TEMPLATE,result=result,disease_colors=DISEASE_COLORS)
        except Exception as e:
            import traceback
            return f"<pre style='color:red;'>Error: {str(e)}\n\n{traceback.format_exc()}</pre>",500

    return render_template_string(HTML_TEMPLATE,disease_colors=DISEASE_COLORS)

@app.route('/download_pdf',methods=['POST'])
def download_pdf():
    import json
    result_json = request.form.get('result_json')
    result=json.loads(result_json)
    pdf_buf=generate_pdf_report(result)
    return send_file(pdf_buf,download_name='agriguard_report.pdf',as_attachment=True)

# === RUN ===
if __name__=='__main__':
    print(f"🚀 AgriGuard running at http://localhost:5000")
    app.run(host='0.0.0.0',port=5000,debug=False)
