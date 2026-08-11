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
# HELPER FUNCTIONS FOR PATIENT & CLINICAL DATA
# ============================================================

def validate_patient_information(name, age, height, height_unit):
    errors = []
    if not name or not name.strip():
        errors.append("Patient Full Name is required and cannot be empty.")
    if age is None or age < 0 or age > 120:
        errors.append("Age must be a valid number between 0 and 120.")
    if height_unit == "cm":
        if height is None or height < 30 or height > 250:
            errors.append("Height must be a valid number between 30 and 250 cm.")
    elif height_unit == "ft/in":
        if not isinstance(height, dict) or "feet" not in height or "inches" not in height:
            errors.append("Height must contain both feet and inches.")
        else:
            ft = height["feet"]
            inc = height["inches"]
            if ft is None or ft < 1 or ft > 8:
                errors.append("Height feet must be a valid number between 1 and 8.")
            if inc is None or inc < 0 or inc > 11:
                errors.append("Height inches must be a valid number between 0 and 11.")
    return errors


def get_disease_information(class_code):
    info = {
        "akiec": {
            "name": "Actinic Keratosis / Intraepithelial Carcinoma (AKIEC)",
            "type": "Potentially Malignant (Precancerous / Early Skin Cancer)",
            "description": "Actinic Keratosis (AK) is a common precancerous skin growth caused by long-term exposure to ultraviolet (UV) radiation. Intraepithelial Carcinoma (Bowen's disease) is an early, superficial form of squamous cell skin cancer.",
            "characteristics": "Typically appears as a rough, dry, scaly, or crusty patch of skin. They may feel rough to the touch (like sandpaper) and range in color from skin-colored to reddish-brown or pink.",
            "concern": "If left untreated, a small percentage of actinic keratoses can progress to invasive squamous cell carcinoma (a type of skin cancer). Thus, professional evaluation is important to assess if treatment is required.",
            "action": "Avoid further direct sun exposure and use daily broad-spectrum sunscreen. Schedule an appointment with a dermatologist or healthcare provider. Treatment options include cryotherapy (freezing), topical creams, or minor removal."
        },
        "bcc": {
            "name": "Basal Cell Carcinoma (BCC)",
            "type": "Malignant (Common Skin Cancer)",
            "description": "Basal Cell Carcinoma is the most common form of skin cancer. It begins in the basal cells, which are found in the outer layer of the skin. BCC is slow-growing and very rarely spreads to other parts of the body.",
            "characteristics": "Commonly presents as a pearly or waxy bump, a flat pinkish growth, a sore that bleeds or does not heal, or a scar-like lesion in sun-exposed areas.",
            "concern": "Although it rarely spreads (metastasizes), it can grow deep into surrounding tissues and bone if left untreated, causing significant localized damage.",
            "action": "A dermatologist should evaluate the lesion. Diagnostic verification is typically done via a minor biopsy. Treatment is highly successful and usually involves surgical excision, scraping, or freezing."
        },
        "bkl": {
            "name": "Benign Keratosis (BKL)",
            "type": "Benign (Non-Cancerous)",
            "description": "Benign Keratosis is a category containing harmless skin growths, most frequently including seborrheic keratosis, solar lentigo (sun spots), or lichen-planus-like keratosis.",
            "characteristics": "These growths often appear to be 'stuck on' the skin. They can be waxy, scaly, or slightly raised, and range in color from light tan to dark brown or black.",
            "concern": "These lesions are completely harmless and are not contagious. However, they can occasionally become irritated, itchy, or rub against clothing. Sometimes, they can look similar to malignant skin conditions, requiring professional confirmation.",
            "action": "No medical intervention is required unless the spot is irritated, bleeds, or is cosmetically undesirable. A dermatologist can easily remove it. If the spot changes shape, color, or grows rapidly, have it checked."
        },
        "df": {
            "name": "Dermatofibroma (DF)",
            "type": "Benign (Non-Cancerous)",
            "description": "Dermatofibroma is a very common, harmless, firm nodule that develops in the deeper layer of the skin, most frequently on the lower legs.",
            "characteristics": "Typically feels like a small, hard bump under the skin (similar to a button). It varies in color from pink to red, brown, or purple. A classic sign is that it 'dimples' inward when pinched from the sides.",
            "concern": "Dermatofibromas are completely benign and do not turn into skin cancer. They are usually asymptomatic but can sometimes be itchy or slightly tender.",
            "action": "Monitoring is standard practice; no treatment is needed unless it causes discomfort, itchiness, or if you prefer it removed. If you notice any sudden changes in size or appearance, consult a doctor."
        },
        "mel": {
            "name": "Melanoma (MEL)",
            "type": "Malignant (Serious Skin Cancer)",
            "description": "Melanoma is a type of skin cancer that develops from pigment-producing cells called melanocytes. It is the most serious form of skin cancer because it can spread to other organs if not caught early.",
            "characteristics": "Often resembles a mole or develops from an existing mole. Warning signs follow the ABCDE rule: Asymmetry, irregular Borders, multiple Colors, large Diameter (>6mm), and Evolving (changing) over time.",
            "concern": "Melanoma is highly aggressive and can spread (metastasize) rapidly. However, when detected and treated early, the survival rate is extremely high.",
            "action": "Seek professional medical evaluation by a dermatologist immediately. Do not delay. A biopsy is typically required to diagnose melanoma. Treatment options include surgical excision, immunotherapy, or targeted therapy."
        },
        "nv": {
            "name": "Melanocytic Nevus (NV)",
            "type": "Benign (Common Mole)",
            "description": "A melanocytic nevus is the medical term for a common mole. It is a benign growth of pigment cells (melanocytes) in the skin.",
            "characteristics": "Generally small, symmetrical, round or oval with smooth, distinct borders. They are usually a uniform color (tan, brown, or dark brown) and can be flat or slightly raised.",
            "concern": "Common moles are benign and harmless. However, individuals with a large number of moles have a higher statistical risk of developing melanoma. A mole that starts changing in appearance requires careful assessment.",
            "action": "Regularly monitor your moles using the ABCDE guidelines. Normal moles do not require treatment. If any mole grows, changes color, bleeds, becomes itchy, or looks very different from your other moles, schedule a dermatologist visit."
        },
        "vasc": {
            "name": "Vascular Lesion (VASC)",
            "type": "Benign (Blood Vessel Growth)",
            "description": "Vascular lesions represent a group of benign growths involving blood vessels. Common types include cherry angiomas, angiokeratomas, and pyogenic granulomas.",
            "characteristics": "Typically appear as bright red, purple, or blue spots or bumps on the skin. They can range from tiny pinhead spots to larger nodules, and cherry angiomas are especially common as people age.",
            "concern": "They are completely non-cancerous and do not pose a risk to health. However, because they are made of blood vessels, they may bleed if bumped, scratched, or irritated.",
            "action": "No treatment is necessary unless they bleed frequently, are physically uncomfortable, or are removed for cosmetic reasons. Protect the area from irritation. Consult a professional if they bleed easily or grow rapidly."
        }
    }
    return info.get(class_code.lower())


def format_patient_summary(name, age, sex, height, height_unit):
    if height_unit == "cm":
        height_str = f"{height:.1f} cm"
    elif height_unit == "ft/in":
        height_str = f"{height['feet']} ft {height['inches']} in"
    else:
        height_str = str(height)
    
    return {
        "Name": name,
        "Age": f"{age} years",
        "Sex": sex,
        "Height": height_str
    }


def generate_clinical_report(
    patient_info,
    prediction_name,
    confidence,
    normalized_entropy,
    trust_score,
    referral,
    top_predictions,
    laplacian_variance,
    sharpness,
    disease_info,
    gradcam_generated=True
):
    referral_text = "CLINICAL REVIEW RECOMMENDED" if referral else "TRUST THRESHOLD MET"
    
    height_unit = patient_info["height_unit"]
    if height_unit == "cm":
        height_str = f"{patient_info['height']:.1f} cm"
    elif height_unit == "ft/in":
        height_str = f"{patient_info['height']['feet']} ft {patient_info['height']['inches']} in"
    else:
        height_str = str(patient_info['height'])

    image_quality_label = "Poor" if sharpness < 0.15 else "Good"
    
    interpretation = (
        "The prediction is below the validated TrustDerma threshold and should be "
        "reviewed by a qualified clinician."
        if referral
        else
        "The prediction meets the validated TrustDerma threshold; clinical "
        "interpretation is still required."
    )
    
    report_text = f"""============================================================
TRUSTDERMA AI-ASSISTED SKIN LESION ASSESSMENT
============================================================

PATIENT INFORMATION
-------------------
Name: {patient_info['name']}
Age: {patient_info['age']} years
Sex: {patient_info['sex']}
Height: {height_str}

IMAGE INFORMATION
-----------------
Image Quality: {image_quality_label}
Sharpness Index: {sharpness:.2%} (Laplacian Var: {laplacian_variance:.2f})

AI ASSESSMENT
-------------
Predicted Condition: {prediction_name}
Model Confidence: {confidence:.2%}
Predictive Uncertainty: {normalized_entropy:.2%}
Trust Score: {trust_score:.2%}
Trust Threshold: {THRESHOLD:.2f}
Assessment Status: {referral_text}

TOP 3 PREDICTIONS
-----------------
1. {top_predictions.iloc[0]['Condition']} — {top_predictions.iloc[0]['Probability']}
2. {top_predictions.iloc[1]['Condition']} — {top_predictions.iloc[1]['Probability']}
3. {top_predictions.iloc[2]['Condition']} — {top_predictions.iloc[2]['Probability']}

CONDITION INFORMATION
---------------------
Condition: {disease_info['name']}
Description: {disease_info['description']}
General Characteristics: {disease_info['characteristics']}
Potential Concern: {disease_info['concern']}
Recommended Next Step: {disease_info['action']}

EXPLAINABILITY
--------------
Grad-CAM++ explanation generated: {"Yes" if gradcam_generated else "No"}

INTERPRETATION
--------------
{interpretation}

IMPORTANT MEDICAL DISCLAIMER
----------------------------
This system is an AI-assisted research/decision-support prototype. The prediction is based on the uploaded image and does not constitute a definitive medical diagnosis. AI predictions may be incorrect. A qualified healthcare professional or dermatologist should make the final clinical assessment. Professional evaluation should be sought especially when a lesion is new, changing, bleeding, painful, rapidly growing, or otherwise concerning.

============================================================
END OF REPORT
============================================================
"""
    return report_text


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

# ============================================================
# 1. PATIENT INFORMATION SECTION
# ============================================================

st.markdown("### 📋 1. Patient Information")

patient_name = st.text_input(
    "Full Name",
    value="",
    placeholder="Enter patient's full name",
    key="patient_name_input"
)

col_age, col_sex = st.columns(2)

with col_age:
    patient_age = st.number_input(
        "Age (Years)",
        min_value=0,
        max_value=120,
        value=30,
        step=1,
        key="patient_age_input"
    )

with col_sex:
    patient_sex = st.selectbox(
        "Sex / Gender",
        options=["Male", "Female", "Other / Prefer not to say"],
        key="patient_sex_input"
    )

height_unit = st.radio(
    "Height Unit",
    options=["cm", "ft/in"],
    horizontal=True,
    key="height_unit_input"
)

if height_unit == "cm":
    patient_height = st.number_input(
        "Height (cm)",
        min_value=30.0,
        max_value=250.0,
        value=170.0,
        step=0.5,
        key="patient_height_cm_input"
    )
else:
    col_ft, col_in = st.columns(2)
    with col_ft:
        ft = st.number_input(
            "Feet",
            min_value=1,
            max_value=8,
            value=5,
            step=1,
            key="patient_height_ft_input"
        )
    with col_in:
        inch = st.number_input(
            "Inches",
            min_value=0,
            max_value=11,
            value=7,
            step=1,
            key="patient_height_in_input"
        )
    patient_height = {"feet": ft, "inches": inch}

patient_errors = validate_patient_information(patient_name, patient_age, patient_height, height_unit)


# ============================================================
# 2. SKIN LESION IMAGE SECTION
# ============================================================

st.divider()
st.markdown("### 📷 2. Skin Lesion Image")
st.write(
    "Upload a clear, well-lit photograph of the skin lesion. "
    "For the most reliable AI analysis, make sure the lesion is visible and in focus."
)

uploaded_file = st.file_uploader(
    "Upload skin lesion photograph",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG and PNG",
    label_visibility="collapsed"
)


if uploaded_file is None:
    st.info("Upload a skin lesion image above to trigger the TrustDerma analysis pipeline.")
    st.stop()


# Read uploaded image
try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception:
    st.error("The uploaded file could not be read as an image. Please check the file format.")
    st.stop()


# Preprocess and calculate quality
tensor, rgb = preprocess(image)
laplacian_variance, sharpness = image_quality(rgb)

# Show uploaded image immediately
st.markdown("#### Uploaded Image Preview")
st.image(image, use_container_width=True)

# Image Quality Alert
quality_poor = sharpness < 0.15
if quality_poor:
    st.warning(
        f"⚠️ **Low Image Quality Warning** (Sharpness Index: {sharpness:.1%})\n\n"
        f"The uploaded image quality appears poor or blurry. AI analysis results "
        f"may be less reliable. For the best clinical decision support, consider "
        f"uploading a sharper, well-focused, and well-lit image."
    )
else:
    st.success(
        f"✓ **Image Quality Check Passed** (Sharpness Index: {sharpness:.1%})"
    )


# Stop execution if patient info contains errors
if patient_errors:
    st.error("Please correct the following errors in the Patient Information section to run the AI assessment:")
    for err in patient_errors:
        st.write(f"- {err}")
    st.stop()


# ============================================================
# RUN MODEL INFERENCE
# ============================================================

with st.spinner("Running TrustDerma AI inference..."):
    (
        probabilities,
        prediction,
        confidence,
        entropy,
        normalized_entropy,
        trust_score,
        referral
    ) = infer(model, device, tensor)

prediction_name = NAMES[CLASSES[prediction]]
disease_info = get_disease_information(CLASSES[prediction])
top_ids = np.argsort(probabilities)[::-1][:3]

top_predictions = pd.DataFrame({
    "Condition": [NAMES[CLASSES[i]] for i in top_ids],
    "Probability": [f"{probabilities[i]:.2%}" for i in top_ids]
})


# ============================================================
# 3. AI ANALYSIS
# ============================================================

st.divider()
st.markdown("### 🧠 3. AI Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="AI-Predicted Condition",
        value=NAMES[CLASSES[prediction]]
    )

with col2:
    st.metric(
        label="Model Confidence",
        value=f"{confidence:.1%}",
        help="The probability percentage assigned by the model to the predicted class."
    )

with col3:
    st.metric(
        label="Trust Score",
        value=f"{trust_score:.1%}",
        help="The calculated reliability score. For the current model, this is equal to the model's confidence."
    )

st.info(
    "ℹ️ **Note on Metrics:** The AI model calculates a probability distribution across the seven target classes. "
    "**Model Confidence** is the probability of the most likely class. The **Trust Score** represents "
    "the reliability rating of the prediction; in this system configuration, the trust score is mathematically "
    "equivalent to the prediction confidence and is compared against a pre-validated 80.0% threshold."
)


# ============================================================
# 4. ASSESSMENT SUMMARY
# ============================================================

st.divider()
st.markdown("### 📋 4. Your Assessment Summary")

summary_data = format_patient_summary(patient_name, patient_age, patient_sex, patient_height, height_unit)
image_quality_label = "Poor" if quality_poor else "Good"
image_quality_color = "#ef4444" if quality_poor else "#22c55e"

if referral:
    action_text = (
        "The AI result does not meet the trust criterion. "
        "A clinical review by a qualified dermatologist is recommended."
    )
    action_color = "#ef4444"
    status_bg = "rgba(239, 68, 68, 0.08)"
    status_border = "#ef4444"
    assessment_status_label = "Referral recommended (Trust threshold not met)"
else:
    action_text = (
        "The AI result meets the current trust criterion. "
        "However, this result is not a diagnosis and should not replace evaluation by a qualified dermatologist."
    )
    action_color = "#22c55e"
    status_bg = "rgba(34, 197, 94, 0.08)"
    status_border = "#22c55e"
    assessment_status_label = "Trust threshold met"

summary_html = f"""
<div style="
    background-color: #ffffff;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    border: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
">
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.5rem;">
        <div>
            <h5 style="margin: 0 0 0.6rem 0; color: #475569; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px;">Patient Details</h5>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Patient Name:</strong> {summary_data['Name']}</p>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Age:</strong> {summary_data['Age']}</p>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Sex:</strong> {summary_data['Sex']}</p>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Height:</strong> {summary_data['Height']}</p>
        </div>
        <div>
            <h5 style="margin: 0 0 0.6rem 0; color: #475569; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px;">AI Assessment</h5>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>AI Prediction:</strong> {prediction_name}</p>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Model Confidence:</strong> {confidence:.1%}</p>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Trust Score:</strong> {trust_score:.1%}</p>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Uncertainty:</strong> {normalized_entropy:.1%}</p>
        </div>
        <div>
            <h5 style="margin: 0 0 0.6rem 0; color: #475569; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px;">Technical Quality</h5>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Image Quality:</strong> <span style="color: {image_quality_color}; font-weight: 700;">{image_quality_label}</span></p>
            <p style="margin: 0.3rem 0; color: #1e293b; font-size: 0.95rem;"><strong>Sharpness Index:</strong> {sharpness:.1%}</p>
        </div>
    </div>
    <div style="
        margin-top: 1.2rem;
        padding: 1rem;
        background-color: {status_bg};
        border-left: 4px solid {status_border};
        border-radius: 6px;
    ">
        <h6 style="margin: 0; color: #1e293b; font-size: 0.95rem; font-weight: 700;">Recommended Action:</h6>
        <p style="margin: 0.3rem 0 0 0; color: #334155; font-size: 0.9rem; line-height: 1.45;">{action_text}</p>
    </div>
</div>
"""
st.markdown(summary_html, unsafe_allow_html=True)


# ============================================================
# 5. UNDERSTANDING THE PREDICTION
# ============================================================

st.divider()
st.markdown("### 🔍 5. Understanding the Prediction")

st.info(
    f"**AI Prediction Statement:** The AI model predicts that the uploaded image is most consistent with "
    f"**{prediction_name}** (classification probability of {confidence:.1%}). **This result is not a diagnosis.**"
)

st.markdown(f"#### 🧬 About the Condition: {disease_info['name']}")
st.write(f"**Classification Category:** `{disease_info['type']}`")
st.write(disease_info['description'])

col_char, col_concern = st.columns(2)

with col_char:
    st.markdown("#### 🔬 Common Characteristics")
    st.write(disease_info['characteristics'])

with col_concern:
    st.markdown("#### ⚠️ Why it may require medical attention")
    st.write(disease_info['concern'])

st.markdown("#### 🩺 Recommended Next Steps")
st.write(disease_info['action'])

st.warning(
    "⚠️ **Important Safety Disclaimer:** AI-assisted assessments are decision-support prototypes and can be incorrect. "
    "You should seek professional medical evaluation by a qualified dermatologist or medical practitioner "
    "especially when a lesion is new, changing color, changing size, bleeding, painful, rapidly growing, "
    "or otherwise concerning."
)


# ============================================================
# 6. TOP PREDICTIONS
# ============================================================

st.divider()
st.markdown("### 📊 6. Top Predictions")
st.write("The table below details the top three classification categories predicted by the model:")

st.dataframe(
    top_predictions,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# 7. UNCERTAINTY & TRUST
# ============================================================

st.divider()
st.markdown("### 🛡️ 7. Uncertainty & Trust")

st.markdown("#### Model Predictive Uncertainty")
st.progress(float(max(0, min(1, 1 - normalized_entropy))))
st.caption(
    f"Normalized Predictive Entropy: **{normalized_entropy:.4f}** (or {normalized_entropy:.1%}). "
    f"A value close to 0% indicates the model is highly certain about its primary prediction, "
    f"while a higher value indicates the prediction probabilities are spread across multiple classes."
)

st.markdown("#### Trust Decision Analysis")
if referral:
    st.error(
        f"⚠️ **Clinical Review Recommended**\n\n"
        f"**Trust Score:** {trust_score:.2%} | **Validated Threshold:** {THRESHOLD:.2%}\n\n"
        f"The prediction does not meet the validated trust criterion. It is recommended that this case "
        f"be referred for manual clinical review."
    )
else:
    st.success(
        f"✓ **Trust Threshold Met**\n\n"
        f"**Trust Score:** {trust_score:.2%} | **Validated Threshold:** {THRESHOLD:.2%}\n\n"
        f"The prediction meets the validated trust threshold. Clinical interpretation is still required."
    )


# ============================================================
# 8. GRAD-CAM++ EXPLAINABILITY
# ============================================================

st.divider()
st.markdown("### 🔥 8. Grad-CAM++ Explainability")
st.write(
    "Grad-CAM++ highlights regions of the image that contributed to the model's prediction. "
    "This visualization is intended to help understand the model's focus and should not be "
    "interpreted as a medical diagnostic map."
)

gradcam_success = False
with st.spinner("Generating visual explanation..."):
    try:
        gradcam_image = generate_gradcam(
            model,
            device,
            tensor,
            prediction,
            rgb
        )
        gradcam_success = True

        cam_left, cam_right = st.columns(2)
        with cam_left:
            st.image(image, caption="Original Image", use_container_width=True)
        with cam_right:
            st.image(
                gradcam_image,
                caption=f"Grad-CAM++ Explanation Heatmap (Focusing on: {prediction_name})",
                use_container_width=True
            )
    except Exception as error:
        st.warning("Grad-CAM++ heatmap visualization could not be generated.")
        st.exception(error)


# ============================================================
# 9. CLINICAL-STYLE REPORT
# ============================================================

st.divider()
st.markdown("### 📋 9. Clinical-Style Report")
st.write("A professional clinical report summary is generated below. You can download this report for your files.")

report_data = {
    "name": patient_name,
    "age": patient_age,
    "sex": patient_sex,
    "height": patient_height,
    "height_unit": height_unit
}

report_text = generate_clinical_report(
    patient_info=report_data,
    prediction_name=prediction_name,
    confidence=confidence,
    normalized_entropy=normalized_entropy,
    trust_score=trust_score,
    referral=referral,
    top_predictions=top_predictions,
    laplacian_variance=laplacian_variance,
    sharpness=sharpness,
    disease_info=disease_info,
    gradcam_generated=gradcam_success
)

st.code(report_text, language="text")

st.download_button(
    label="⬇️ Download Clinical Report",
    data=report_text,
    file_name=f"TrustDerma_Report_{patient_name.replace(' ', '_')}.txt" if patient_name.strip() else "TrustDerma_Clinical_Report.txt",
    mime="text/plain"
)


# ============================================================
# 10. TECHNICAL DETAILS
# ============================================================

with st.expander("🔬 10. Technical Details"):
    st.json({
        "device": str(device),
        "model": "tf_efficientnetv2_s",
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
        "huggingface_model": f"{HF_REPO}/{HF_FILENAME}"
    })