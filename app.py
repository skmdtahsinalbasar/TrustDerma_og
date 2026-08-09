import os, io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2, torch, timm
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.set_page_config(page_title="TrustDerma", page_icon="🩺", layout="wide")

CLASSES = ["akiec","bcc","bkl","df","mel","nv","vasc"]
NAMES = {
 "akiec":"Actinic Keratoses / Intraepithelial Carcinoma",
 "bcc":"Basal Cell Carcinoma","bkl":"Benign Keratosis",
 "df":"Dermatofibroma","mel":"Melanoma",
 "nv":"Melanocytic Nevus","vasc":"Vascular Lesion"
}
MODEL_PATH = os.getenv("TRUSTDERMA_MODEL","models/best_baseline.pth")
THRESHOLD = 0.80

transform = A.Compose([A.Resize(384,384), A.Normalize(), ToTensorV2()])

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("tf_efficientnetv2_s", pretrained=False, num_classes=7)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.to(device).eval()
    return model, device

def preprocess(img):
    rgb = np.array(img.convert("RGB"))
    x = transform(image=rgb)["image"].unsqueeze(0)
    return x, rgb

def infer(model, device, x):
    with torch.no_grad():
        p = F.softmax(model(x.to(device)), dim=1).cpu().numpy()[0]
    pred = int(p.argmax())
    conf = float(p[pred])
    ent = float(-np.sum(p*np.log(p+1e-8)))
    norm_ent = ent/np.log(7)
    trust = conf
    return p,pred,conf,ent,norm_ent,trust,trust < THRESHOLD

def quality(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    lap=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    return lap, float(1-np.exp(-lap/1000))

def gradcam(model,device,x,pred,rgb):
    cam=GradCAMPlusPlus(model=model,target_layers=[model.conv_head])
    heat=cam(input_tensor=x.to(device),
             targets=[ClassifierOutputTarget(pred)])[0]
    h,w=rgb.shape[:2]
    heat_r=cv2.resize(heat,(w,h))
    return show_cam_on_image(rgb.astype(np.float32)/255,heat_r,use_rgb=True)

st.markdown("""
<style>
.block-container{max-width:1400px;padding-top:1.5rem}
.hero{padding:1.5rem 1.7rem;border-radius:20px;background:linear-gradient(135deg,#102a43,#167d9a);color:white;margin-bottom:1.2rem}
.hero h1{margin:0;font-size:2.5rem}.hero p{margin:.4rem 0 0;opacity:.9}
</style>
<div class="hero"><h1>🩺 TrustDerma</h1>
<p>AI-assisted skin lesion analysis • confidence • uncertainty • trust • explainability</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("TrustDerma")
    st.write("**EfficientNetV2-S**")
    st.write("**Input:** 384 × 384")
    st.write("**Classes:** 7")
    st.write(f"**Locked trust threshold:** {THRESHOLD:.2f}")
    st.warning("Research/decision-support prototype. Not a definitive diagnosis.")

try:
    model,device=load_model()
except Exception as e:
    st.error(str(e))
    st.info("Put your trained best_baseline.pth inside the models/ folder.")
    st.stop()

file=st.file_uploader("Upload a skin lesion image",type=["jpg","jpeg","png"])
if not file:
    st.info("Upload a clear JPG/PNG image to start the complete TrustDerma pipeline.")
    st.stop()

img=Image.open(file).convert("RGB")
x,rgb=preprocess(img)

with st.spinner("Running TrustDerma analysis..."):
    probs,pred,conf,ent,norm_ent,trust,referral=infer(model,device,x)
    lap,sharp=quality(rgb)

st.success("Analysis complete.")

a,b,c,d=st.columns(4)
a.metric("Prediction",NAMES[CLASSES[pred]])
b.metric("Confidence",f"{conf:.1%}")
c.metric("Trust Score",f"{trust:.1%}")
d.metric("Uncertainty",f"{norm_ent:.1%}")

if referral:
    st.error(f"⚠️ CLINICAL REVIEW RECOMMENDED — trust {trust:.2%} < {THRESHOLD:.2f}")
else:
    st.success(f"✓ TRUST THRESHOLD MET — trust {trust:.2%} ≥ {THRESHOLD:.2f}")

l,r=st.columns(2)
with l:
    st.subheader("Uploaded Image")
    st.image(img,use_container_width=True)
with r:
    st.subheader("Top 3 Predictions")
    ids=np.argsort(probs)[::-1][:3]
    st.dataframe(pd.DataFrame({
        "Condition":[NAMES[CLASSES[i]] for i in ids],
        "Probability":[f"{probs[i]:.2%}" for i in ids]
    }),hide_index=True,use_container_width=True)
    st.subheader("Uncertainty")
    st.progress(float(1-norm_ent))
    st.caption(f"Normalized predictive entropy: {norm_ent:.4f}")

st.divider()
st.header("Grad-CAM++ Explainability")
with st.spinner("Generating explanation..."):
    overlay=gradcam(model,device,x,pred,rgb)
g1,g2=st.columns(2)
g1.image(img,caption="Original",use_container_width=True)
g2.image(overlay,caption=f"Grad-CAM++ — {NAMES[CLASSES[pred]]}",use_container_width=True)

st.divider()
st.header("Clinical-Style Report")
report=f"""TRUSTDERMA AI SKIN LESION REPORT

Prediction: {NAMES[CLASSES[pred]]}
Confidence: {conf:.2%}
Predictive Uncertainty: {norm_ent:.2%}
Trust Score: {trust:.2%}
Locked Trust Threshold: {THRESHOLD:.2f}
Referral Status: {"CLINICAL REVIEW RECOMMENDED" if referral else "NO AUTOMATIC REFERRAL"}

Top Predictions:
1. {NAMES[CLASSES[ids[0]]]} — {probs[ids[0]]:.2%}
2. {NAMES[CLASSES[ids[1]]]} — {probs[ids[1]]:.2%}
3. {NAMES[CLASSES[ids[2]]]} — {probs[ids[2]]:.2%}

Image Sharpness Index: {sharp:.2%}

Interpretation:
{"The prediction is below the validated TrustDerma threshold and should be reviewed by a qualified clinician."
 if referral else
 "The prediction meets the validated TrustDerma threshold; clinical interpretation is still required."}

IMPORTANT:
This is a research/decision-support prototype. It does not establish a
definitive medical diagnosis and must not replace professional medical
assessment, dermoscopy, biopsy, or clinical judgment.
"""
st.code(report,language="text")
st.download_button("Download Clinical Report",report,"TrustDerma_Clinical_Report.txt","text/plain")

with st.expander("Technical details"):
    st.json({"device":str(device),"model":"EfficientNetV2-S","input":"384x384",
             "trust_formulation":"Confidence","threshold":THRESHOLD,
             "confidence":conf,"normalized_entropy":norm_ent,
             "trust_score":trust,"referral":referral,
             "laplacian_variance":lap})
