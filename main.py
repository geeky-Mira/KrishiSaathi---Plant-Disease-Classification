import os
import io
import json
import re
import time
import contextlib

import streamlit as st
from PIL import Image
import numpy as np
from google import genai
from google.genai import types, errors
from gtts import gTTS
from tensorflow.lite.python.interpreter import Interpreter

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### AI-powered plant disease detection system for sustainable agriculture"
    }
)

# ── Session state defaults ──────────────────────────────────────────────────
if 'prediction' not in st.session_state:
    st.session_state.update({
        "prediction": {},
        "advice": "",
        "audio_bytes": None,
        "title": "",
        "lang_code": "en",
        "lang_name": "English",
        "processing": False,
        "word_limit": 100,    # default word limit
        "inference_time": None
    })

def clear_results():
    st.session_state.prediction = {}
    st.session_state.advice = ""
    st.session_state.audio_bytes = None
    st.session_state.title = ""
    st.session_state.inference_time = None

# ── Load TFLite interpreter (with optional delegate) ─────────────────────────
@st.cache_resource
def load_interpreter():
    interpreter = Interpreter(model_path="plant_disease_model_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

# ── Load class labels ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_class_names():
    with open("class_names.json", "r") as f:
        return json.load(f)

# ── TTS helper (suppress logs) ───────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_tts_audio(text: str, lang: str) -> bytes:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        buf = io.BytesIO()
        gTTS(text=text, lang=lang).write_to_fp(buf)
        buf.seek(0)
        return buf.getvalue()

# ── Silent wrapper ──────────────────────────────────────────────────────────
def silent_get_tts_audio(text, lang):
    return get_tts_audio(text, lang)

# ── Init models & client ────────────────────────────────────────────────────
interpreter    = load_interpreter()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
class_names    = load_class_names()
client         = genai.Client(api_key=os.getenv("GEMINI_API_KEY"), vertexai=False)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🌱 Plant Disease Detection")
    mode = st.radio("Page", ["🏠 Home", "🔍 Disease Recognition"])

# ── Image preprocessing ─────────────────────────────────────────────────────
@st.cache_data(max_entries=3)
def preprocess_image(image, input_details):
    img = Image.open(image).convert("RGB")
    target_size = tuple(input_details[0]["shape"][1:3][::-1])
    img = img.resize(target_size)
    arr = np.array(img)
    dtype = input_details[0]["dtype"]
    scale, zp = input_details[0]["quantization"]
    if dtype == np.uint8 and scale != 0:
        q = (arr / scale + zp).astype(np.uint8)
        return np.expand_dims(q, 0)
    return np.expand_dims(arr.astype(dtype), 0)

# ── Enhanced prompt builder ─────────────────────────────────────────────────
def build_prompt(plant: str, disease: str, lang_code: str, word_limit: int) -> str:
    sys_instr = {
        "en": "You are an expert plant pathologist. Provide clear, concise advice for farmers.",
        "bn": "আপনি একজন অভিজ্ঞ উদ্ভিদ রোগ বিশেষজ্ঞ। কৃষকদের জন্য পরিষ্কার, সংক্ষিপ্ত পরামর্শ দিন।"
    }
    limit_txt = {
        "en": f"Respond in no more than {word_limit} words.",
        "bn": f"{word_limit} শব্দের বেশি না।"
    }
    if disease.lower() == "healthy":
        examples = {
            "en": """Example for Tomato:
here are some tips for maintaining good health of the plant:
• Water deeply 2-3 times weekly
• Apply balanced fertilizer every 4 weeks
• Prune for air circulation
• Monitor early pest signs""",
            "bn": """টমেটো উদাহরণ:
গাছের সুস্বাস্থ্য বজায় রাখার জন্য এখানে কিছু টিপস দেওয়া হল:
• সপ্তাহে ২-৩ বার জল দিন
• প্রতি ৪ সপ্তাহে সার প্রয়োগ
• বাতাসের জন্য ছাঁটাই
• পোকার লক্ষণ পর্যবেক্ষণ"""
        }
        title = {"en":"Maintenance Guide","bn":"রক্ষণাবেক্ষণ নির্দেশিকা"}[lang_code]
        prompt = (
            f"{sys_instr[lang_code]}\n"
            f"{limit_txt[lang_code]}\n"
            f"Provide maintenance tips for healthy {plant}.\n"
            f"{examples[lang_code]}"
        )
    else:
        examples = {
            "en": """Example for Tomato Blight:
Causes:
• Fungal spores in wet conditions
• Poor air flow
Remedies:
• Apply copper fungicide
• Remove infected leaves""",
            "bn": """টমেটো ব্লাইট উদাহরণ:
কারণ:
• আর্দ্রতায় ছত্রাক
• বাতাসের অভাব
প্রতিকার:
• তামার ছত্রাকনাশক
• আক্রান্ত পাতা সরান"""
        }
        title = {"en":"Disease Analysis","bn":"রোগ বিশ্লেষণ"}[lang_code]
        prompt = (
            f"{sys_instr[lang_code]}\n"
            f"{limit_txt[lang_code]}\n"
            f"Explain {disease} in {plant}.\n"
            f"{examples[lang_code]}"
        )
    st.session_state.title = title
    return prompt

# ── Home page ───────────────────────────────────────────────────────────────
if mode == "🏠 Home":
    _, col, _ = st.columns([2,5,1])
    with col:
        st.image("AI-agriculture.jpg", width=400)
    st.markdown("""
    <h1 style='text-align:center;color:green;'>🌿 Plant Disease Detection System</h1>
    <p style='text-align:center;'>AI-driven disease recognition for sustainable farming.</p>
    <hr>
    <div style='max-width:800px;margin:0 auto;'>
      <h3>📌 How to Use</h3>
      <ol>
        <li>Select <strong>Disease Recognition</strong></li>
        <li>Upload or capture a <strong>leaf image</strong></li>
        <li>Pick your <strong>language</strong> & <strong>word limit</strong></li>
        <li>Click <strong>🔎 Predict Disease</strong></li>
        <li>Click <strong>🔊 Play Explanation</strong></li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

# ── Disease Recognition page ────────────────────────────────────────────────
else:
    st.header("🌱 Disease Recognition")
    method = st.radio("Select Input Method", ["Upload Image","Use Camera"], horizontal=True)
    if method == "Upload Image":
        test_image = st.file_uploader("📸 Choose Image", type=["jpg","jpeg","png"],
                                      on_change=clear_results, key="uploader")
    else:
        test_image = st.camera_input("📷 Take Picture", on_change=clear_results, key="camera")

    if test_image:
        # Language & word limit
        langs = {"English":"en","বাংলা":"bn"}
        c1, c2 = st.columns(2)
        with c1:
            new_lang = st.selectbox("Select Language", list(langs.keys()),
                                    index=list(langs.values()).index(st.session_state.lang_code),
                                    on_change=clear_results)
            st.session_state.lang_name = new_lang
            st.session_state.lang_code = langs[new_lang]
        with c2:
            wl = st.slider("Word Limit", min_value=100, max_value=500,
                           value=st.session_state.word_limit, step=10, on_change=clear_results)
            st.session_state.word_limit = wl

    if test_image:
        c_img, c_txt = st.columns([1,3])
        with c_img:
            st.image(test_image, caption="Leaf", width=200)

        if c_txt.button("🔎 Predict Disease", key="predict"):
            status = st.info("🔍 Predicting…")
            st.session_state.processing = True
            try:
                clear_results()
                # 1) Inference timing
                t0 = time.time()
                arr = preprocess_image(test_image, input_details)
                interpreter.set_tensor(input_details[0]["index"], arr)
                interpreter.invoke()
                out = interpreter.get_tensor(output_details[0]["index"])
                idx = np.argmax(out)
                t1 = time.time()

                # Record prediction
                label = class_names[idx]
                plant, disease = (label.split("___",1) if "___" in label else (label, "Healthy"))
                disease = disease.replace("_"," ")
                st.session_state.prediction = {"plant":plant, "disease":disease}
                st.session_state.inference_time = t1 - t0

                # 2) Build prompt & generate advice
                prompt = build_prompt(plant, disease,
                                      st.session_state.lang_code,
                                      st.session_state.word_limit)
                with st.spinner("💡 Generating advice…"):
                    resp = client.models.generate_content(
                        model="gemini-1.5-flash-latest",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            top_k=1,
                            top_p=1.0,
                            max_output_tokens=int(st.session_state.word_limit * 1.8)  # approx tokens
                        )
                    )
                    raw = resp.text.strip()

                # 3) TTS
                speech = re.sub(r"[•\*\-]+","",raw)
                speech = re.sub(r"\s{2,}"," ",speech).strip()
                st.session_state.audio_bytes = silent_get_tts_audio(
                    speech, st.session_state.lang_code
                )
                st.session_state.advice = raw

            except errors.ServerError:
                st.error("⚠️ Server busy—try again later.")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                status.empty()
                st.session_state.processing = False

    # ── Display results ─────────────────────────────────────────────────────
    if st.session_state.prediction and not st.session_state.processing:
        pred = st.session_state.prediction
        with c_txt:
            # Show inference time
            st.write(f"⏱ Prediction time: **{st.session_state.inference_time:.2f}s**")

            # Healthy vs diseased layout
            if pred["disease"].lower() == "healthy":
                msg = (
                    f"🌿 Your **{pred['plant']}** looks healthy!"
                    if st.session_state.lang_code=="en"
                    else f"🌿 আপনার **{pred['plant']}** গাছ সুস্থ!"
                )
                st.success(msg)
                st.markdown(f"### {st.session_state.title}")
                for line in st.session_state.advice.split("\n"):
                    if line.strip():
                        st.markdown(f"- {line.strip()}")
            else:
                tpl = (
                    f"🌿 Plant: **{pred['plant']}**   🦠 Disease: **{pred['disease']}**"
                    if st.session_state.lang_code=="en"
                    else f"🌿 গাছ: **{pred['plant']}**   🦠 রোগ: **{pred['disease']}**"
                )
                st.success(tpl)
                st.markdown(f"### {st.session_state.title}")
                for ln in st.session_state.advice.split("\n"):
                    if ln.strip(): st.markdown(ln)

            # Play audio
            btn = (
                "🔊 Play Explanation" if st.session_state.lang_code=="en"
                else "🔊 ব্যাখ্যা শুনুন"
            )
            if st.button(btn, key="play"):
                st.audio(st.session_state.audio_bytes, format="audio/mp3")
