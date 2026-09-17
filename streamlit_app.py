import os
import time
import zipfile
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
import torchvision
from torchvision.transforms import functional as F
from ultralytics import YOLO

# ==============================================================================
# 1. إعدادات الصفحة وهوية المنظومة (Page Configuration & CSS Theme)
# ==============================================================================
st.set_page_config(
    page_title="SafeSight AI | Industrial PPE Platform",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-title { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-num { font-size: 26px; font-weight: 800; margin: 4px 0; }
    .color-safe { color: #10b981; }
    .color-warn { color: #f59e0b; }
    .color-crit { color: #ef4444; }
    .color-info { color: #38bdf8; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. مدير الأوزان الذكي وفك الضغط التلقائي (Automated Weight & Zip Resolver)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def extract_zip_if_exists(zip_name: str, target_extracted_file: str) -> str:
    """البحث عن ملف zip وفك ضغطه تلقائياً إذا لم يكن الملف المستخرج متوفراً"""
    extracted_path = os.path.join(MODELS_DIR, target_extracted_file)
    if os.path.exists(extracted_path):
        return extracted_path

    # البحث عن ملف الـ zip في مجلد models أو المسار الرئيسي
    possible_zips = [
        os.path.join(MODELS_DIR, zip_name),
        os.path.join(BASE_DIR, zip_name),
        zip_name
    ]
    
    for z_path in possible_zips:
        if os.path.exists(z_path):
            try:
                with zipfile.ZipFile(z_path, 'r') as zip_ref:
                    zip_ref.extractall(MODELS_DIR)
                if os.path.exists(extracted_path):
                    return extracted_path
            except Exception as e:
                st.sidebar.error(f"Error extracting {zip_name}: {e}")
                
    return ""

def locate_weight(file_name: str) -> str:
    """البحث الديناميكي عن ملف الوزن المباشر"""
    candidates = [
        os.path.join(MODELS_DIR, file_name),
        os.path.join(BASE_DIR, file_name),
        file_name
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""

# فحص وفك ضغط أوزان RetinaNet تلقائياً إن وجدت
retina_weight_path = extract_zip_if_exists("retinanet_ppe_best.zip", "retinanet_ppe_best.pth")
if not retina_weight_path:
    retina_weight_path = locate_weight("retinanet_ppe_best.pth")

MODEL_REGISTRY = {
    "YOLO11s (Attention Enhanced - Best Accuracy)": {
        "file": locate_weight("best.pt"),
        "type": "ultralytics",
        "fallback": "yolov8n.pt"
    },
    "YOLOv8n (Lightweight - Best Edge FPS)": {
        "file": locate_weight("best8n.pt"),
        "type": "ultralytics",
        "fallback": "yolov8n.pt"
    },
    "RetinaNet (ResNet-50 Baseline)": {
        "file": retina_weight_path,
        "type": "torchvision",
        "fallback": None
    }
}

@st.cache_resource(show_spinner=False)
def load_model_instance(model_name: str):
    """تحميل النموذج وتخزينه في الكاش لمنع استهلاك الذاكرة وتكرار التحميل"""
    meta = MODEL_REGISTRY.get(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if meta["type"] == "ultralytics":
        weight_to_load = meta["file"] if (meta["file"] and os.path.exists(meta["file"])) else meta["fallback"]
        model = YOLO(weight_to_load)
        return {"model": model, "type": "ultralytics", "device": device}
        
    elif meta["type"] == "torchvision":
        # بناء معمارية RetinaNet مع 6 كلاسات (خلفية + 5 فئات)
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None, 
            weights_backbone=None, 
            num_classes=6
        )
        if meta["file"] and os.path.exists(meta["file"]):
            state_dict = torch.load(meta["file"], map_location=device)
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return {"model": model, "type": "torchvision", "device": device}

# ==============================================================================
# 3. المنطق الرياضي للربط التشريحي وتدقيق السلامة (Compliance Logic)
# ==============================================================================
def verify_anatomical_region(gear_box, person_box, part="head") -> bool:
    """التحقق الهندسي من وقوع المعدة في موضعها التشريحي الصحيح من جسم العامل"""
    gx_c = (gear_box[0] + gear_box[2]) / 2.0
    gy_c = (gear_box[1] + gear_box[3]) / 2.0
    px1, py1, px2, py2 = person_box
    p_h = py2 - py1

    if not (px1 <= gx_c <= px2):
        return False

    if part == "head":
        return py1 <= gy_c <= (py1 + 0.42 * p_h)
    elif part == "torso":
        return (py1 + 0.20 * p_h) <= gy_c <= (py1 + 0.85 * p_h)
    return False

def resolve_gear_conflict(gear_list, person_box, part: str):
    """حسم التعارض بين الكشوفات المتضاربة واعتماد الأعلى ثقة"""
    matched = [g for g in gear_list if verify_anatomical_region(g['box'], person_box, part)]
    if not matched:
        return "missing", 0.0
    selected = max(matched, key=lambda item: item['conf'])
    return selected['label'], selected['conf']

def annotate_safety_scene(img_bgr, persons, helmets, vests):
    """رسم الصناديق وتوليد تقرير السلامة المنطقي"""
    annotated = img_bgr.copy()
    safe_n, warn_n, crit_n = 0, 0, 0
    audit_table = []

    for idx, p in enumerate(persons):
        p_box = p['box']
        px1, py1, px2, py2 = p_box

        h_stat, h_conf = resolve_gear_conflict(helmets, p_box, "head")
        v_stat, v_conf = resolve_gear_conflict(vests, p_box, "torso")

        has_helmet = (h_stat == "helmet")
        has_vest = (v_stat == "vest")

        if has_helmet and has_vest:
            safe_n += 1
            box_color = (0, 200, 0)
            status_desc = "Safe"
            display_tag = f"Worker #{idx+1} [SAFE]"
        elif has_helmet or has_vest:
            warn_n += 1
            box_color = (0, 140, 255)
            status_desc = "Warning"
            missing_text = "NO VEST" if has_helmet else "NO HELMET"
            display_tag = f"Worker #{idx+1} [{missing_text}]"
        else:
            crit_n += 1
            box_color = (0, 0, 255)
            status_desc = "Critical"
            display_tag = f"Worker #{idx+1} [CRITICAL]"

        cv2.rectangle(annotated, (px1, py1), (px2, py2), box_color, 3)
        cv2.putText(annotated, display_tag, (px1, max(18, py1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2, cv2.LINE_AA)

        audit_table.append({
            "Worker ID": f"Worker #{idx+1}",
            "Detector Conf": f"{p['conf']:.2f}",
            "Hardhat State": f"✅ Equipped ({h_conf:.2f})" if has_helmet else "❌ Missing",
            "Safety Vest State": f"✅ Equipped ({v_conf:.2f})" if has_vest else "❌ Missing",
            "Compliance Status": status_desc
        })

    for h in helmets:
        hx1, hy1, hx2, hy2 = h['box']
        hc = (255, 120, 0) if h['label'] == 'helmet' else (0, 0, 255)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), hc, 2)

    for v in vests:
        vx1, vy1, vx2, vy2 = v['box']
        vc = (0, 220, 220) if v['label'] == 'vest' else (0, 0, 255)
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), vc, 2)

    return annotated, audit_table, safe_n, warn_n, crit_n

def predict_any_model(img_bgr, model_dict, p_conf, g_conf):
    """محرك استنتاج موحد يدعم YOLO و Torchvision RetinaNet"""
    start_t = time.perf_counter()
    person_group, helmet_group, vest_group = [], [], []
    
    m_instance = model_dict["model"]
    m_type = model_dict["type"]
    device = model_dict["device"]

    if m_type == "ultralytics":
        results = m_instance.predict(source=img_bgr, conf=min(p_conf, g_conf), verbose=False)[0]
        names = m_instance.names
        for b in results.boxes:
            coords = b.xyxy[0].cpu().numpy().astype(int)
            score = float(b.conf[0].cpu().numpy())
            c_name = names[int(b.cls[0].cpu().numpy())].lower()

            if c_name == 'person' and score >= p_conf:
                person_group.append({'box': coords, 'conf': score})
            elif c_name in ['helmet', 'no_helmet'] and score >= g_conf:
                helmet_group.append({'box': coords, 'conf': score, 'label': c_name})
            elif c_name in ['vest', 'no_vest'] and score >= g_conf:
                vest_group.append({'box': coords, 'conf': score, 'label': c_name})

    elif m_type == "torchvision":
        retina_map = {1: 'helmet', 2: 'no_helmet', 3: 'no_vest', 4: 'person', 5: 'vest'}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = m_instance(tensor)[0]
            
        for i in range(len(preds['boxes'])):
            score = float(preds['scores'][i].cpu().numpy())
            cls_idx = int(preds['labels'][i].cpu().numpy())
            coords = preds['boxes'][i].cpu().numpy().astype(int)
            
            if cls_idx in retina_map:
                c_name = retina_map[cls_idx]
                if c_name == 'person' and score >= p_conf:
                    person_group.append({'box': coords, 'conf': score})
                elif c_name in ['helmet', 'no_helmet'] and score >= g_conf:
                    helmet_group.append({'box': coords, 'conf': score, 'label': c_name})
                elif c_name in ['vest', 'no_vest'] and score >= g_conf:
                    vest_group.append({'box': coords, 'conf': score, 'label': c_name})

    latency_ms = (time.perf_counter() - start_t) * 1000.0
    ann_img, audit_rows, s_cnt, w_cnt, c_cnt = annotate_safety_scene(
        img_bgr, person_group, helmet_group, vest_group
    )
    return ann_img, audit_rows, len(person_group), s_cnt, w_cnt, c_cnt, latency_ms

# ==============================================================================
# 4. بناء واجهة المستخدم الرسومية والتبويبات (Dashboard UI)
# ==============================================================================
st.title("🦺 SafeSight AI — PPE Compliance Monitor")
st.caption("AI-Powered Safety Equipment Monitoring System (YOLO11, YOLOv8 & RetinaNet)")

with st.sidebar:
    st.subheader("⚙️ Architecture & Sensitivity")
    active_engine = st.selectbox("Active Model Backbone", list(MODEL_REGISTRY.keys()))
    
    # إشعار بحالة ملفات الأوزان
    meta_info = MODEL_REGISTRY[active_engine]
    if meta_info["type"] == "torchvision" and not meta_info["file"]:
        st.warning("⚠️ RetinaNet weights file not found. Running with initialized baseline weights.")
    elif meta_info["type"] == "ultralytics" and not meta_info["file"]:
        st.info("ℹ️ Custom fine-tuned weights missing. Using standard pre-trained fallback.")
        
    st.markdown("---")
    st.markdown("**Threshold Calibration**")
    p_slider = st.slider("Person Confidence Threshold", 0.20, 0.90, 0.50, 0.05)
    g_slider = st.slider("Safety Gear Threshold", 0.15, 0.85, 0.35, 0.05)
    st.info("💡 **Scientific Basis:** Gear Threshold (0.35) matches the F1-Confidence peak (0.371) to ensure high recall for small objects.")

tabs = st.tabs([
    "🔍 Site Inspector", 
    "⚡ Dual Model Compare", 
    "📊 Benchmarks & Complexity", 
    "📈 Diagnostic Artifacts"
])

# ----------------------------------------------------
# TAB 1: الفحص المباشر (Site Inspector)
# ----------------------------------------------------
with tabs[0]:
    in_col, out_col = st.columns([1, 2])
    with in_col:
        st.subheader("Input Stream")
        file_input = st.file_uploader("Upload Inspection Image", type=["jpg", "jpeg", "png"], key="main_input")
        execute_trigger = st.button("Run Compliance Scan", type="primary", use_container_width=True)

    if file_input is not None:
        raw_arr = np.asarray(bytearray(file_input.read()), dtype=np.uint8)
        loaded_bgr = cv2.imdecode(raw_arr, cv2.IMREAD_COLOR)

        if execute_trigger:
            active_model_dict = load_model_instance(active_engine)
            res_bgr, records, tot, s_num, w_num, c_num, elapsed = predict_any_model(
                loaded_bgr, active_model_dict, p_slider, g_slider
            )

            with out_col:
                st.subheader("Analyzed Frame Output")
                st.image(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("### 📡 Live Compliance Telemetry")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.markdown(f'<div class="metric-card"><div class="metric-title">Total Detected</div><div class="metric-num">{tot}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="metric-card"><div class="metric-title">Fully Safe</div><div class="metric-num color-safe">{s_num}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="metric-card"><div class="metric-title">Warnings</div><div class="metric-num color-warn">{w_num}</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="metric-card"><div class="metric-title">Critical Violations</div><div class="metric-num color-crit">{c_num}</div></div>', unsafe_allow_html=True)
            k5.markdown(f'<div class="metric-card"><div class="metric-title">Latency</div><div class="metric-num color-info">{elapsed:.1f} ms</div></div>', unsafe_allow_html=True)

            if records:
                st.markdown("### 📋 Individual Worker Audit Log")
                st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        with out_col:
            st.info("Awaiting image upload to start automated safety analysis.")

# ----------------------------------------------------
# TAB 2: المقارنة الثنائية المتزامنة (Dual Compare)
# ----------------------------------------------------
with tabs[1]:
    st.subheader("Simultaneous Multi-Model Benchmarking")
    c_left, c_right = st.columns(2)
    with c_left:
        m1_choice = st.selectbox("Primary Model (Left)", list(MODEL_REGISTRY.keys()), index=0)
    with c_right:
        m2_choice = st.selectbox("Comparison Baseline (Right)", list(MODEL_REGISTRY.keys()), index=2)

    dual_upload = st.file_uploader("Upload Frame for Side-by-Side Comparison", type=["jpg", "jpeg", "png"], key="dual_input")
    if dual_upload and st.button("Execute Comparative Run", use_container_width=True):
        d_bytes = np.asarray(bytearray(dual_upload.read()), dtype=np.uint8)
        dual_img = cv2.imdecode(d_bytes, cv2.IMREAD_COLOR)

        m1_net = load_model_instance(m1_choice)
        m2_net = load_model_instance(m2_choice)

        out1, _, tot1, safe1, _, crit1, lat1 = predict_any_model(dual_img, m1_net, p_slider, g_slider)
        out2, _, tot2, safe2, _, crit2, lat2 = predict_any_model(dual_img, m2_net, p_slider, g_slider)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{m1_choice}** — `{lat1:.1f} ms`")
            st.image(cv2.cvtColor(out1, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detected: {tot1} | Safe: {safe1} | Critical Violations: {crit1}")
        with col2:
            st.markdown(f"**{m2_choice}** — `{lat2:.1f} ms`")
            st.image(cv2.cvtColor(out2, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detected: {tot2} | Safe: {safe2} | Critical Violations: {crit2}")

# ----------------------------------------------------
# TAB 3: المقاييس الحسابية والتعقيد (Benchmarks)
# ----------------------------------------------------
with tabs[2]:
    st.subheader("Theoretical Complexity & Edge Feasibility")
    benchmark_df = pd.DataFrame([
        {
            "Architecture": "YOLO11s (Attention Enhanced)", 
            "Parameters (M)": 9.4, 
            "GFLOPs": 21.4, 
            "Est. FPS (GPU)": 62, 
            "mAP@50": "92.0%", 
            "Deployment Role": "High Accuracy Central Server Monitoring"
        },
        {
            "Architecture": "YOLOv8n (Lightweight Edge)", 
            "Parameters (M)": 3.2, 
            "GFLOPs": 8.7, 
            "Est. FPS (GPU)": 132, 
            "mAP@50": "90.55%", 
            "Deployment Role": "Real-Time Embedded Edge & IoT Cameras"
        },
        {
            "Architecture": "RetinaNet (ResNet-50 Baseline)", 
            "Parameters (M)": 34.0, 
            "GFLOPs": 90.0, 
            "Est. FPS (GPU)": 12, 
            "mAP@50": "81.0%", 
            "Deployment Role": "Academic Baseline (Too Heavy for Real-time Video)"
        }
    ])
    st.dataframe(benchmark_df, use_container_width=True)

# ----------------------------------------------------
# TAB 4: الأدلة التشخيصية (Diagnostic Artifacts)
# ----------------------------------------------------
with tabs[3]:
    st.subheader("Empirical Training Validation Proofs")
    
    def display_artifact(filename_list, caption_text):
        search_dirs = [BASE_DIR, os.path.join(BASE_DIR, "docs"), os.path.join(BASE_DIR, "evaluation_output")]
        found_path = None
        for d in search_dirs:
            for fname in filename_list:
                cand = os.path.join(d, fname)
                if os.path.exists(cand):
                    found_path = cand
                    break
            if found_path:
                break
        
        if found_path:
            st.image(found_path, use_container_width=True)
            st.caption(caption_text)
        else:
            st.warning(f"Artifact {filename_list[0]} not found in workspace.")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### Normalized Confusion Matrix")
        display_artifact(
            ["confusion_matrix_normalized.png", "confusion_matrix_normalized_2.png", "confusion_matrix_normalized_3.png"],
            "Class-wise classification precision and background confusion distribution."
        )
    with p2:
        st.markdown("#### Precision-Recall (PR) Curve")
        display_artifact(
            ["BoxPR_curve.png", "BoxPR_curve_2.png", "BoxPR_curve_3.png"],
            "Area Under Curve (AUC) demonstrating mAP performance across all 5 classes."
        )

    st.markdown("---")
    f1_c, loss_c = st.columns(2)
    with f1_c:
        st.markdown("#### F1-Confidence Optimization Curve")
        display_artifact(
            ["BoxF1_curve.png", "BoxF1_curve_2.png", "BoxF1_curve_3.png"],
            "Empirical mathematical validation for setting the default gear threshold to 0.35."
        )
    with loss_c:
        st.markdown("#### Training Loss History")
        display_artifact(
            ["results.jpg", "results.png"],
            "Decay of bounding box, classification, and DFL losses over training epochs."
        )
