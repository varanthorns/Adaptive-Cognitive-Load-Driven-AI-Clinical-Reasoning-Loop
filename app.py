import streamlit as st
import json, random, pandas as pd, time
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_mic_recorder import mic_recorder

# ===================== CONFIG =====================
st.set_page_config(layout="wide", page_title="ACLR Ultimate Clinical Reasoning")

# ===================== EVALUATION LOGIC =====================
def evaluate_base(dx, reasoning, case, profession):
    """ฟังก์ชันพื้นฐานสำหรับคำนวณคะแนน Dx, Evidence และ Logic"""
    target = case.get("interprofessional_answers", {}).get(profession, case.get("answer", ""))
    
    # 1. Accuracy Score (5 pts)
    try:
        vec = TfidfVectorizer().fit_transform([str(dx).lower(), str(target).lower()])
        sim = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    except: sim = 0
    dx_score = 5 if sim > 0.75 else (3 if sim > 0.45 else 0)
    level = "correct" if sim > 0.75 else ("close" if sim > 0.45 else "wrong")

    # 2. Key Evidence Score (3 pts)
    found_keys = [k for k in case.get("key_points", []) if k.lower() in reasoning.lower()]
    r_score = min(3, len(found_keys))

    # 3. Clinical Logic Score (2 pts)
    logic_words = ["because", "therefore", "thus", "due to", "เนื่องจาก", "ดังนั้น", "ทำให้", "ส่งผล"]
    d_score = 2 if any(w in reasoning.lower() for w in logic_words) else 0

    return dx_score, r_score, d_score, target, found_keys, level

def evaluate_pro(dx, reasoning, case, profession, confidence, selected_ddx):
    """ฟังก์ชันประเมินผลระดับสูง รวมระบบ Safety และ Confidence"""
    dx_s, r_s, d_s, target, used, level = evaluate_base(dx, reasoning, case, profession)
    
    # แก้ไข Syntax Error จากเวอร์ชันก่อนหน้า
    total_base = dx_s + r_s + d_s

    # 1. DDx Safety Check (
