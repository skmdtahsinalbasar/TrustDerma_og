import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import torch
import timm
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from huggingface_hub import hf_hub_download

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TrustDerma",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# TRUSTDERMA CONFIGURATION
# ============================================================

CLASSES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

NAMES = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevus",
    "vasc": "Vascular Lesion"
}

# Final validated TrustDerma threshold
THRESHOLD = 0.80

# Hugging Face model repository
HF_REPO = "tahsinalbsr/TrustDerma-best-model"
HF_FILENAME = "best_baseline.pth"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2rem 2.2rem;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #102a43,
            #167d9a
        );
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.7rem;
        font-weight: 800;
    }

    .hero p {
        margin-top: .5rem;
        font-size: 1.05rem;
        opacity: .92;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 1rem;
    }

    .trust-box {
        padding: 1.2rem;
        border-radius: 15px;
        background: #f5f8fa;
        border: 1px solid #dce5eb;
        margin: .5rem 0 1rem 0;
    }

    .small-muted {
        color: #6b7280;
        font-size: .9rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🩺 TrustDerma</h1>
        <p>
            Trusted AI for skin lesion analysis —
            confidence • uncertainty • trust • explainability
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# MODEL DOWNLOAD
# ============================================================

@st.cache_resource(show_spinner=False)
def download_model():

    st.write("Preparing TrustDerma AI model...")

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILENAME,
        repo_type="model"
    )

    return model_path


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading TrustDerma AI model...")
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_path = download_model()

    model = timm.create_model(
        "tf_efficientnetv2_s",
        pretrained=False,
        num_classes=7
    )

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device


# ============================================================
# PREPROCESSING
# ============================================================

transform = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2()
    ]
)


def preprocess(image):

    rgb = np.array(
        image.convert("RGB")
    )

    tensor = transform(
        image=rgb
    )["image"]

    tensor = tensor.unsqueeze(0)

    return tensor, rgb


# ============================================================
# MODEL INFERENCE
# ============================================================

def infer(model, device, tensor):

    with torch.no_grad():

        logits = model(
            tensor.to(device)
        )

        probabilities = F.softmax(
            logits,
            dim=1
        ).cpu().numpy()[0]

    prediction = int(
        probabilities.argmax()
    )

    confidence = float(
        probabilities[prediction]
    )

    # Predictive entropy
    entropy = float(
        -np.sum(
            probabilities *
            np.log(probabilities + 1e-8)
        )
    )

    # Normalize entropy to [0,1]
    normalized_entropy = (
        entropy / np.log(len(CLASSES))
    )

    # --------------------------------------------------------
    # FINAL TRUSTDERMA TRUST FORMULATION
    #
    # Your final experiment reports:
    # Trust formulation : Confidence
    # Locked threshold   : 0.80
    #
    # Therefore:
    # Trust Score = Prediction Confidence
    # --------------------------------------------------------

    trust_score = confidence

    referral = (
        trust_score < THRESHOLD
    )

    return (
        probabilities,
        prediction,
        confidence,
        entropy,
        normalized_entropy,
        trust_score,
        referral
    )


# ============================================================
# IMAGE QUALITY
# ============================================================

def image_quality(rgb):

    gray = cv2.cvtColor(
        rgb,
        cv2.COLOR_RGB2GRAY
    )

    laplacian_variance = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    )

    sharpness = float(
        1 - np.exp(
            -laplacian_variance / 1000
        )
    )

    return (
        laplacian_variance,
        sharpness
    )


# ============================================================
# GRAD-CAM++
# ============================================================

def generate_gradcam(
    model,
    device,
    tensor,
    prediction,
    rgb
):

    target_layers = [
        model.conv_head
    ]

    cam = GradCAMPlusPlus(
        model=model,
        target_layers=target_layers
    )

    grayscale_cam = cam(
        input_tensor=tensor.to(device),
        targets=[
            ClassifierOutputTarget(
                prediction
            )
        ]
    )[0]

    height, width = rgb.shape[:2]

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (width, height)
    )

    overlay = show_cam_on_image(
        rgb.astype(np.float32) / 255.0,
        grayscale_cam,
        use_rgb=True
    )

    return overlay


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🩺 TrustDerma")

    st.write(
        "**Model:** EfficientNetV2-S"
    )

    st.write(
        "**Input:** 384 × 384"
    )

    st.write(
        "**Classes:** 7"
    )

    st.write(
        f"**Validated Trust Threshold:** {THRESHOLD:.2f}"
    )

    st.divider()

    st.caption(
        "TrustDerma is a research and "
        "clinical decision-support prototype."
    )

    st.warning(
        "This system does not establish "
        "a definitive medical diagnosis."
    )


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, device = load_model()

except Exception as error:

    st.error(
        "Unable to load the TrustDerma model."
    )

    st.exception(error)

    st.stop()


# ============================================================
# UPLOAD
# ============================================================

st.header("Upload Skin Lesion Image")

uploaded_file = st.file_uploader(
    "Upload a clear skin lesion image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ],
    help="Supported formats: JPG, JPEG and PNG"
)


if uploaded_file is None:

    st.info(
        "Upload an image to start the TrustDerma analysis pipeline."
    )

    st.stop()


# ============================================================
# READ IMAGE
# ============================================================

try:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception:

    st.error(
        "The uploaded file could not be read as an image."
    )

    st.stop()


tensor, rgb = preprocess(image)


# ============================================================
# RUN COMPLETE PIPELINE
# ============================================================

with st.spinner(
    "Running TrustDerma analysis..."
):

    (
        probabilities,
        prediction,
        confidence,
        entropy,
        normalized_entropy,
        trust_score,
        referral
    ) = infer(
        model,
        device,
        tensor
    )

    laplacian_variance, sharpness = (
        image_quality(rgb)
    )


st.success(
    "TrustDerma analysis completed."
)


# ============================================================
# MAIN RESULT
# ============================================================

prediction_name = NAMES[
    CLASSES[prediction]
]


st.subheader("AI Prediction")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Prediction",
        prediction_name
    )

with col2:

    st.metric(
        "Confidence",
        f"{confidence:.1%}"
    )

with col3:

    st.metric(
        "Trust Score",
        f"{trust_score:.1%}"
    )

with col4:

    st.metric(
        "Uncertainty",
        f"{normalized_entropy:.1%}"
    )


# ============================================================
# TRUST DECISION
# ============================================================

if referral:

    st.error(
        f"⚠️ CLINICAL REVIEW RECOMMENDED\n\n"
        f"Trust Score: {trust_score:.2%}\n\n"
        f"Validated Threshold: {THRESHOLD:.2f}"
    )

else:

    st.success(
        f"✓ TRUST THRESHOLD MET\n\n"
        f"Trust Score: {trust_score:.2%}\n\n"
        f"Validated Threshold: {THRESHOLD:.2f}"
    )


# ============================================================
# IMAGE + TOP PREDICTIONS
# ============================================================

left, right = st.columns(
    [1, 1]
)


with left:

    st.subheader(
        "Uploaded Image"
    )

    st.image(
        image,
        use_container_width=True
    )


with right:

    st.subheader(
        "Top 3 Predictions"
    )

    top_ids = np.argsort(
        probabilities
    )[::-1][:3]

    top_predictions = pd.DataFrame(
        {
            "Condition": [
                NAMES[CLASSES[i]]
                for i in top_ids
            ],
            "Probability": [
                f"{probabilities[i]:.2%}"
                for i in top_ids
            ]
        }
    )

    st.dataframe(
        top_predictions,
        hide_index=True,
        use_container_width=True
    )

    st.subheader(
        "Predictive Uncertainty"
    )

    st.progress(
        float(
            max(
                0,
                min(
                    1,
                    1 - normalized_entropy
                )
            )
        )
    )

    st.caption(
        f"Normalized predictive entropy: "
        f"{normalized_entropy:.4f}"
    )


# ============================================================
# GRAD-CAM++
# ============================================================

st.divider()

st.header(
    "🔥 Grad-CAM++ Explainability"
)

with st.spinner(
    "Generating visual explanation..."
):

    try:

        gradcam_image = generate_gradcam(
            model,
            device,
            tensor,
            prediction,
            rgb
        )

        cam_left, cam_right = st.columns(2)

        with cam_left:

            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        with cam_right:

            st.image(
                gradcam_image,
                caption=(
                    "Grad-CAM++ — "
                    + prediction_name
                ),
                use_container_width=True
            )

    except Exception as error:

        st.warning(
            "Grad-CAM++ could not be generated."
        )

        st.exception(error)


# ============================================================
# CLINICAL-STYLE REPORT
# ============================================================

st.divider()

st.header(
    "📋 TrustDerma Clinical-Style Report"
)


referral_text = (
    "CLINICAL REVIEW RECOMMENDED"
    if referral
    else
    "TRUST THRESHOLD MET"
)


interpretation = (
    "The prediction is below the validated "
    "TrustDerma threshold and should be "
    "reviewed by a qualified clinician."
    if referral
    else
    "The prediction meets the validated "
    "TrustDerma threshold; clinical "
    "interpretation is still required."
)


report = f"""
============================================================
TRUSTDERMA AI SKIN LESION REPORT
============================================================

Model
------------------------------------------------------------
Architecture: EfficientNetV2-S
Input Size: 384 × 384
Number of Classes: 7

Prediction
------------------------------------------------------------
Predicted Condition: {prediction_name}
Confidence: {confidence:.2%}

Trust & Uncertainty
------------------------------------------------------------
Trust Formulation: Confidence
Trust Score: {trust_score:.2%}
Validated Trust Threshold: {THRESHOLD:.2f}
Normalized Predictive Entropy: {normalized_entropy:.2%}
Referral Status: {referral_text}

Top Predictions
------------------------------------------------------------
1. {NAMES[CLASSES[top_ids[0]]]} — {probabilities[top_ids[0]]:.2%}
2. {NAMES[CLASSES[top_ids[1]]]} — {probabilities[top_ids[1]]:.2%}
3. {NAMES[CLASSES[top_ids[2]]]} — {probabilities[top_ids[2]]:.2%}

Image Quality
------------------------------------------------------------
Laplacian Variance: {laplacian_variance:.2f}
Sharpness Index: {sharpness:.2%}

Interpretation
------------------------------------------------------------
{interpretation}

IMPORTANT
------------------------------------------------------------
TrustDerma is a research and clinical
decision-support prototype.

This output does not establish a definitive
medical diagnosis and must not replace:

• Professional medical assessment
• Dermoscopic examination
• Histopathological examination / biopsy
• Clinical judgment

============================================================
END OF REPORT
============================================================
"""


st.code(
    report,
    language="text"
)


st.download_button(
    label="⬇️ Download Clinical Report",
    data=report,
    file_name="TrustDerma_Clinical_Report.txt",
    mime="text/plain"
)


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander(
    "🔬 Technical Details"
):

    st.json(
        {
            "device": str(device),
            "model": "EfficientNetV2-S",
            "input_size": "384x384",
            "classes": 7,
            "trust_formulation": "Confidence",
            "validated_threshold": THRESHOLD,
            "confidence": confidence,
            "trust_score": trust_score,
            "normalized_entropy": normalized_entropy,
            "referral": referral,
            "laplacian_variance": laplacian_variance,
            "sharpness_index": sharpness,
            "huggingface_model": (
                f"{HF_REPO}/"
                f"{HF_FILENAME}"
            )
        }
    )