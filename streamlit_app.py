import os
import time
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
# 2. إدارة الأوزان والمسارات المستقرة (Model Registry & Weights)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def locate_file(rel_path: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, rel_path),
        os.path.join(BASE_DIR, "Safety_Helmet-Reflective_Jacket", rel_path),
        os.path.join(MODELS_DIR, os.path.basename(rel_path)),
        os.path.join(BASE_DIR, os.path.basename(rel_path))
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""

MODEL_REGISTRY = {
    "YOLO11s (Attention Enhanced - Best Accuracy)": {
        "file": locate_file("models/best.pt"),
        "type": "ultralytics",
        "fallback": "yolov8n.pt"
    },
    "YOLOv8n (Lightweight Edge - Best FPS)": {
        "file": locate_file("models/best8n.pt"),
        "type": "ultralytics",
        "fallback": "yolov8n.pt"
    },
    "RetinaNet (ResNet-50 Baseline)": {
        "file": locate_file("models/retinanet_ppe_best.pth"),
        "type": "torchvision",
        "fallback": None
    }
}

@st.cache_resource(show_spinner=False)
def load_detection_engine(model_name: str):
    meta = MODEL_REGISTRY.get(model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if meta["type"] == "ultralytics":
        weight_path = meta["file"] if (meta["file"] and os.path.exists(meta["file"])) else meta["fallback"]
        model = YOLO(weight_path)
        return {"model": model, "type": "ultralytics", "device": device, "has_weights": bool(meta["file"])}
        
    elif meta["type"] == "torchvision":
        # تصحيح عدد الكلاسات ليتوافق مع تدريب RetinaNet في ريزنت 50 (5 كلاسات + خلفية = 6)
        model = torchvision.models.detection.retinanet_resnet50_fpn(
            weights=None, 
            weights_backbone=None, 
            num_classes=6
        )
        has_w = False
        if meta["file"] and os.path.exists(meta["file"]):
            state_dict = torch.load(meta["file"], map_location=device)
            model.load_state_dict(state_dict)
            has_w = True
        model.to(device)
        model.eval()
        return {"model": model, "type": "torchvision", "device": device, "has_weights": has_w}

# ==============================================================================
# 3. المنطق التشريحي وقواعد السلامة الصناعية الصارمة
# ==============================================================================
def verify_anatomical_bounds(gear_box, person_box, region="head") -> bool:
    gx_c = (gear_box[0] + gear_box[2]) / 2.0
    gy_c = (gear_box[1] + gear_box[3]) / 2.0
    px1, py1, px2, py2 = person_box
    p_h = py2 - py1

    if not (px1 <= gx_c <= px2):
        return False

    if region == "head":
        return py1 <= gy_c <= (py1 + 0.38 * p_h)
    elif region == "torso":
        return (py1 + 0.20 * p_h) <= gy_c <= (py1 + 0.85 * p_h)
    return False

def extract_best_gear(gear_list, person_box, region: str):
    candidates = [g for g in gear_list if verify_anatomical_bounds(g['box'], person_box, region)]
    if not candidates:
        return "none", 0.0
    best_candidate = max(candidates, key=lambda x: x['conf'])
    return best_candidate['label'], best_candidate['conf']

def draw_styled_tag(img, text, x, y, bg_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thick = 2
    (t_w, t_h), _ = cv2.getTextSize(text, font, scale, thick)
    y1 = max(0, y - t_h - 10)
    y2 = max(t_h + 10, y)
    cv2.rectangle(img, (x, y1), (x + t_w + 10, y2), bg_color, -1)
    cv2.putText(img, text, (x + 5, y2 - 5), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def evaluate_worker_compliance(img_bgr, persons, helmets, vests):
    annotated = img_bgr.copy()
    safe_cnt, warn_cnt, crit_cnt = 0, 0, 0
    audit_data = []

    for idx, p in enumerate(persons):
        p_box = p['box']
        px1, py1, px2, py2 = p_box

        h_label, h_conf = extract_best_gear(helmets, p_box, "head")
        v_label, v_conf = extract_best_gear(vests, p_box, "torso")

        has_helmet = (h_label == "helmet")
        has_vest = (v_label == "vest")

        # 1. Fully Safe: لابس الخوذة والسترة الفسفورية معاً
        # 2. Critical: مش لابس الاتنين معاً
        # 3. Warning: لابس واحدة ومش لابس التانية
        if has_helmet and has_vest:
            safe_cnt += 1
            box_color = (0, 200, 0)
            status_text = "Safe"
            tag_label = f"Worker #{idx+1} [FULLY SAFE]"
        elif (not has_helmet) and (not has_vest):
            crit_cnt += 1
            box_color = (0, 0, 255)
            status_text = "Critical"
            tag_label = f"Worker #{idx+1} [CRITICAL]"
        else:
            warn_cnt += 1
            box_color = (0, 140, 255)
            status_text = "Warning"
            missing_cause = "NO VEST" if has_helmet else "NO HELMET"
            tag_label = f"Worker #{idx+1} [WARN: {missing_cause}]"

        cv2.rectangle(annotated, (px1, py1), (px2, py2), box_color, 3)
        draw_styled_tag(annotated, tag_label, px1, py1, box_color)

        audit_data.append({
            "Worker ID": f"Worker #{idx+1}",
            "Person Conf": f"{p['conf']:.2f}",
            "Hardhat": f"✅ Helmet ({h_conf:.2f})" if has_helmet else ("❌ Missing" if h_label == "no_helmet" else "❓ Undetected"),
            "Safety Vest": f"✅ Reflective Vest ({v_conf:.2f})" if has_vest else ("❌ Missing" if v_label == "no_vest" else "❓ Undetected"),
            "Compliance State": status_text
        })

    for h in helmets:
        hx1, hy1, hx2, hy2 = h['box']
        c = (255, 120, 0) if h['label'] == 'helmet' else (0, 0, 255)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), c, 2)

    for v in vests:
        vx1, vy1, vx2, vy2 = v['box']
        c = (0, 220, 220) if v['label'] == 'vest' else (0, 0, 255)
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), c, 2)

    return annotated, audit_data, safe_cnt, warn_cnt, crit_cnt

def run_inference_pipeline(img_bgr, model_package, p_conf, g_conf, v_strict_conf):
    start_time = time.perf_counter()
    person_list, helmet_list, vest_list = [], [], []

    m = model_package["model"]
    m_type = model_package["type"]
    dev = model_package["device"]

    if m_type == "ultralytics":
        results = m.predict(source=img_bgr, conf=0.15, verbose=False)[0]
        names = m.names
        for b in results.boxes:
            box = b.xyxy[0].cpu().numpy().astype(int)
            conf = float(b.conf[0].cpu().numpy())
            label = names[int(b.cls[0].cpu().numpy())].lower()

            if label == 'person' and conf >= p_conf:
                person_list.append({'box': box, 'conf': conf})
            elif label in ['helmet', 'no_helmet'] and conf >= g_conf:
                helmet_list.append({'box': box, 'conf': conf, 'label': label})
            elif label == 'vest' and conf >= v_strict_conf:
                vest_list.append({'box': box, 'conf': conf, 'label': label})
            elif label == 'no_vest' and conf >= g_conf:
                vest_list.append({'box': box, 'conf': conf, 'label': label})

    elif m_type == "torchvision":
        if not model_package["has_weights"]:
            return img_bgr, [], 0, 0, 0, 0, 0.0

        # خريطة كلاسات RetinaNet الصحيحة والمفصولة تماماً
        retina_map = {1: 'helmet', 2: 'no_helmet', 3: 'no_vest', 4: 'person', 5: 'vest'}
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        tensor = F.to_tensor(img_rgb).unsqueeze(0).to(dev)
        with torch.no_grad():
            preds = m(tensor)[0]

        for i in range(len(preds['boxes'])):
            conf = float(preds['scores'][i].cpu().numpy())
            cls_id = int(preds['labels'][i].cpu().numpy())
            box = preds['boxes'][i].cpu().numpy().astype(int)

            if cls_id in retina_map:
                label = retina_map[cls_id]
                if label == 'person' and conf >= p_conf:
                    person_list.append({'box': box, 'conf': conf})
                elif label in ['helmet', 'no_helmet'] and conf >= g_conf:
                    helmet_list.append({'box': box, 'conf': conf, 'label': label})
                elif label == 'vest' and conf >= v_strict_conf:
                    vest_list.append({'box': box, 'conf': conf, 'label': label})
                elif label == 'no_vest' and conf >= g_conf:
                    vest_list.append({'box': box, 'conf': conf, 'label': label})

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    ann_img, audit_table, safe_n, warn_n, crit_n = evaluate_worker_compliance(
        img_bgr, person_list, helmet_list, vest_list
    )
    return ann_img, audit_table, len(person_list), safe_n, warn_n, crit_n, latency_ms

# ==============================================================================
# 4. بناء واجهة المستخدم
# ==============================================================================
st.title("🦺 SafeSight AI — PPE Compliance Monitor")
st.caption("AI-Powered Real-Time Safety Equipment Monitoring System (YOLO11, YOLOv8 & RetinaNet)")

with st.sidebar:
    st.subheader("⚙️ Architecture & Sensitivity")
    active_engine = st.selectbox("Active Model Backbone", list(MODEL_REGISTRY.keys()))
    
    st.markdown("---")
    st.markdown("**Calibrated Threshold Controls**")
    p_slider = st.slider("Person Confidence Threshold", 0.20, 0.90, 0.50, 0.05)
    g_slider = st.slider("Safety Gear Threshold", 0.15, 0.85, 0.35, 0.05)
    vest_slider = st.slider(
        "Strict Reflective Vest Threshold", 0.30, 0.95, 0.65, 0.05,
        help="Calibrated to 0.65+ to strictly reject casual clothes, t-shirts, and dark jackets lacking high-vis reflective bands."
    )
    
    st.info("💡 **Scientific Basis:** Gear Threshold (0.35) matches the F1-Confidence peak (0.371) to ensure high recall for small objects, while Vest Threshold (0.65) rejects casual clothes.")

tabs = st.tabs([
    "🔍 Site Inspector", 
    "⚡ Dual Model Compare", 
    "📊 Benchmarks & Complexity", 
    "📈 Diagnostic Artifacts"
])

# ----------------------------------------------------
# TAB 1: الفحص الفردي المباشر
# ----------------------------------------------------
with tabs[0]:
    in_col, out_col = st.columns([1, 2])
    with in_col:
        st.subheader("Input Stream")
        file_input = st.file_uploader("Upload Inspection Image", type=["jpg", "jpeg", "png"], key="main_input")
        execute_btn = st.button("Run Compliance Scan", type="primary", use_container_width=True)

    if file_input is not None:
        raw_arr = np.asarray(bytearray(file_input.read()), dtype=np.uint8)
        loaded_bgr = cv2.imdecode(raw_arr, cv2.IMREAD_COLOR)

        if execute_btn:
            active_pack = load_detection_engine(active_engine)
            res_bgr, records, tot, s_num, w_num, c_num, elapsed = run_inference_pipeline(
                loaded_bgr, active_pack, p_slider, g_slider, vest_slider
            )

            with out_col:
                st.subheader("Analyzed Frame Output")
                st.image(cv2.cvtColor(res_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown("### 📡 Live Compliance Telemetry")
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.markdown(f'<div class="metric-card"><div class="metric-title">Total Workers</div><div class="metric-num">{tot}</div></div>', unsafe_allow_html=True)
            k2.markdown(f'<div class="metric-card"><div class="metric-title">Fully Safe</div><div class="metric-num color-safe">{s_num}</div></div>', unsafe_allow_html=True)
            k3.markdown(f'<div class="metric-card"><div class="metric-title">Warnings</div><div class="metric-num color-warn">{w_num}</div></div>', unsafe_allow_html=True)
            k4.markdown(f'<div class="metric-card"><div class="metric-title">Critical Violations</div><div class="metric-num color-crit">{c_num}</div></div>', unsafe_allow_html=True)
            k5.markdown(f'<div class="metric-card"><div class="metric-title">Inference Speed</div><div class="metric-num color-info">{elapsed:.1f} ms</div></div>', unsafe_allow_html=True)

            if records:
                st.markdown("### 📋 Individual Worker Audit Log")
                st.dataframe(pd.DataFrame(records), use_container_width=True)
    else:
        with out_col:
            st.info("Awaiting image upload to start automated safety analysis.")

# ----------------------------------------------------
# TAB 2: المقارنة المزدوجة المتزامنة
# ----------------------------------------------------
with tabs[1]:
    st.subheader("Simultaneous Multi-Model Benchmarking")
    c_left, c_right = st.columns(2)
    with c_left:
        m1_choice = st.selectbox("Primary Model (Left)", list(MODEL_REGISTRY.keys()), index=0)
    with c_right:
        m2_choice = st.selectbox("Comparison Baseline (Right)", list(MODEL_REGISTRY.keys()), index=1)

    dual_upload = st.file_uploader("Upload Frame for Side-by-Side Comparison", type=["jpg", "jpeg", "png"], key="dual_input")
    if dual_upload and st.button("Execute Comparative Run", use_container_width=True):
        d_bytes = np.asarray(bytearray(dual_upload.read()), dtype=np.uint8)
        dual_img = cv2.imdecode(d_bytes, cv2.IMREAD_COLOR)

        m1_pack = load_detection_engine(m1_choice)
        m2_pack = load_detection_engine(m2_choice)

        out1, _, tot1, safe1, _, crit1, lat1 = run_inference_pipeline(dual_img, m1_pack, p_slider, g_slider, vest_slider)
        out2, _, tot2, safe2, _, crit2, lat2 = run_inference_pipeline(dual_img, m2_pack, p_slider, g_slider, vest_slider)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{m1_choice}** — `{lat1:.1f} ms`")
            st.image(cv2.cvtColor(out1, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detected: {tot1} | Fully Safe: {safe1} | Critical Violations: {crit1}")
        with col2:
            st.markdown(f"**{m2_choice}** — `{lat2:.1f} ms`")
            st.image(cv2.cvtColor(out2, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Detected: {tot2} | Fully Safe: {safe2} | Critical Violations: {crit2}")

# ----------------------------------------------------
# TAB 3: المقاييس النظرية والتعقيد الحسابي
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
# TAB 4: الأدلة التشخيصية المعزولة بدقة لكل نموذج
# ----------------------------------------------------
with tabs[3]:
    st.subheader("Empirical Training Validation Proofs")
    
    proof_model = st.selectbox(
        "Select Architecture to Display Proofs",
        ["YOLO11s (Attention Enhanced)", "YOLOv8n (Lightweight)", "RetinaNet (Baseline)"]
    )

    # تحديد المجلد المباشر والصحيح بناءً على هيكلة ملفات المشروع
    if "11s" in proof_model:
        subfolder_name = "evaluation_output"
    elif "v8n" in proof_model:
        subfolder_name = "evaluation8n_output"
    else:
        subfolder_name = "evaluation_retinanet_output"

    TARGET_METRICS_DIR = os.path.join(
        BASE_DIR, "SafeSight-Workspace", "frontend", "public", subfolder_name, "test_metrics"
    )
    ALT_METRICS_DIR = os.path.join(
        BASE_DIR, "frontend", "public", subfolder_name, "test_metrics"
    )

    def get_proof_path(filename: str) -> str:
        for folder in [TARGET_METRICS_DIR, ALT_METRICS_DIR]:
            p = os.path.join(folder, filename)
            if os.path.exists(p):
                return p
        return ""

    if "RetinaNet" in proof_model:
        # موديل RetinaNet يحتوي على رسم الخسارة results.png فقط
        st.markdown("#### Training Loss & Validation History")
        loss_path = get_proof_path("results.png")
        if loss_path:
            st.image(loss_path, use_container_width=True)
            st.caption("Training loss decay and validation divergence curves for RetinaNet (ResNet-50).")
        else:
            st.info("Results plot for RetinaNet not found.")
    else:
        # نماذج YOLO (تحتوي على الرسوم البيانية الأربعة الكاملة)
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("#### Normalized Confusion Matrix")
            m_path = get_proof_path("confusion_matrix_normalized.png")
            if m_path:
                st.image(m_path, use_container_width=True)
                st.caption(f"Class-wise classification precision for {proof_model}.")
            else:
                st.info("Matrix image not available.")

        with p2:
            st.markdown("#### Precision-Recall (PR) Curve")
            pr_path = get_proof_path("BoxPR_curve.png")
            if pr_path:
                st.image(pr_path, use_container_width=True)
                st.caption(f"Precision-Recall AUC curve for {proof_model}.")
            else:
                st.info("PR curve not available.")

        st.markdown("---")
        f1_c, loss_c = st.columns(2)
        with f1_c:
            st.markdown("#### F1-Confidence Optimization Curve")
            f1_path = get_proof_path("BoxF1_curve.png")
            if f1_path:
                st.image(f1_path, use_container_width=True)
                st.caption(f"F1-Confidence tradeoff curve for {proof_model}.")
            else:
                st.info("F1 curve not available.")

        with loss_c:
            if "11s" in proof_model:
                st.markdown("#### Training Loss History")
                l_path = get_proof_path("results.png")
                if l_path:
                    st.image(l_path, use_container_width=True)
                    st.caption("Decay of bounding box and classification losses over 50 epochs.")
                else:
                    st.info("Loss plot not available.")
            else:
                # YOLOv8n يحتوي على صور الـ val_batch بدلاً من results.png
                st.markdown("#### Test Validation Batch Predictions")
                val_path = get_proof_path("val_batch0_pred.jpg")
                if val_path:
                    st.image(val_path, use_container_width=True)
                    st.caption("Ground-truth vs Model Predictions on Test Batch (YOLOv8n).")
                else:
                    st.info("Batch prediction image not available.")
