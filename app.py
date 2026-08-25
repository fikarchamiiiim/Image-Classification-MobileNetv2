import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from model import get_mobilenetv2_model

# ================== KONFIGURASI HALAMAN ==================
st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================== DAFTAR KELAS DEFAULT (PlantVillage 38 kelas) ==================
DEFAULT_CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight', 'Grape___healthy',
    'Orange___Haunglongbing', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites', 'Tomato___Target_Spot', 'Tomato___Yellow_Leaf_Curl_Virus',
    'Tomato___mosaic_virus', 'Tomato___healthy'
]

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model(model_path="model.pth", num_classes=38):
    """
    Memuat model MobileNetV2 dari file .pth.
    Mengembalikan model dan daftar kelas jika tersedia di checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_mobilenetv2_model(num_classes=num_classes)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        class_names = None

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            class_names = checkpoint.get('class_names', None)
            st.sidebar.success(f"✅ Loaded checkpoint with {checkpoint.get('num_classes', num_classes)} classes")
        else:
            model.load_state_dict(checkpoint)
            st.sidebar.success("✅ Loaded weights only")

        model = model.to(device)
        model.eval()
        return model, class_names, device
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

# ================== PREPROCESSING ==================
def preprocess_image(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ================== PREDIKSI ==================
def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.squeeze().cpu().numpy()

# ================== CSS CUSTOM ==================
def local_css():
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.8rem;
            font-weight: 700;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #7f8c8d;
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-section {
            border: 2px dashed #27ae60;
            border-radius: 10px;
            padding: 20px;
            background-color: #f0faf0;
        }
        .result-card {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-top: 20px;
        }
        .prediction-label {
            font-size: 1.8rem;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        .confidence {
            font-size: 1.3rem;
            color: #27ae60;
            font-weight: 600;
        }
        .stButton > button {
            background-color: #27ae60;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            border: none;
        }
        .stButton > button:hover {
            background-color: #219a52;
        }
    </style>
    """, unsafe_allow_html=True)

# ================== MAIN APP ==================
def main():
    local_css()

    # Header
    st.markdown('<div class="main-header">🌿 Plant Disease Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Unggah gambar daun tanaman untuk mendeteksi penyakitnya</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Pengaturan Model")
        model_path = st.text_input("Path file model (.pth)", value="model.pth")
        num_classes = st.number_input("Jumlah kelas", min_value=2, max_value=1000, value=38, step=1)

        # Muat model dengan cache
        try:
            model, checkpoint_class_names, device = load_model(model_path, num_classes)
            st.sidebar.success(f"✅ Model dimuat | Device: {device}")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {e}")
            st.stop()

        # Tentukan class names
        if checkpoint_class_names is not None and len(checkpoint_class_names) == num_classes:
            default_names = checkpoint_class_names
            st.sidebar.info("Nama kelas diambil dari checkpoint.")
        else:
            default_names = DEFAULT_CLASS_NAMES[:num_classes]
            st.sidebar.info("Menggunakan daftar nama kelas default (PlantVillage).")

        st.markdown("---")
        st.markdown("**Daftar Kelas** (edit jika perlu)")
        class_names_input = st.text_area(
            "Nama kelas (pisahkan dengan koma)",
            value=", ".join(default_names),
            height=200
        )
        class_names = [name.strip() for name in class_names_input.split(",") if name.strip()]

        if len(class_names) != num_classes:
            st.warning(f"Jumlah nama ({len(class_names)}) ≠ jumlah kelas ({num_classes}). Menggunakan default.")
            class_names = default_names[:num_classes]

    # Layout dua kolom
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📤 Pilih gambar daun...", type=["jpg", "jpeg", "png", "bmp"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)

            if st.button("🔍 Klasifikasikan", type="primary", use_container_width=True):
                with st.spinner("Memproses..."):
                    tensor = preprocess_image(image)
                    probs = predict(model, tensor, device)
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]
                    st.session_state['prediction'] = (pred_idx, confidence, probs)
        else:
            st.info("👆 Silakan unggah gambar terlebih dahulu.")

    with col2:
        st.markdown("### Hasil Prediksi")
        if 'prediction' in st.session_state:
            pred_idx, confidence, probs = st.session_state['prediction']
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else f"Kelas {pred_idx}"

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-label">🏆 {pred_label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">Confidence: {confidence:.2%}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Plot top-5
            st.markdown("#### Top 5 Prediksi")
            top_k = min(5, len(probs))
            top_indices = np.argsort(probs)[-top_k:][::-1]
            top_probs = probs[top_indices]
            top_labels = [class_names[i] if i < len(class_names) else f"Kelas {i}" for i in top_indices]

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.barh(range(top_k), top_probs[::-1], color='#27ae60')
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(top_labels[::-1], fontsize=9)
            ax.set_xlabel('Probabilitas')
            ax.set_title('Distribusi Probabilitas')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Hasil prediksi akan muncul di sini setelah Anda mengunggah gambar dan menekan tombol klasifikasi.")

if __name__ == "__main__":
    main()