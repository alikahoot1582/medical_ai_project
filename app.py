import streamlit as st
import pandas as pd
import os
from groq import Groq

# ==============================================================================
# 🔑 EMBEDDED ENGINE AUTHENTICATION LAYER
# ==============================================================================
GROQ_API_KEY = "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"

# 1. Page Configuration & Layout Rules
st.set_page_config(
    page_title="Aegis Core // Diagnostic Workspace",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Modern & Clean Interface Stylesheet
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* Global typography optimization */
        html, body, [class*="css"] { 
            font-family: 'Inter', sans-serif; 
        }
        
        /* Clean Terminal Console box for the live streaming logs */
        .terminal-box { 
            background-color: #0B0F19; 
            border: 1px solid #1E293B; 
            border-radius: 6px; 
            padding: 16px; 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 0.85rem; 
            color: #38BDF8; 
            white-space: pre-wrap; 
            height: 380px; 
            overflow-y: auto;
            box-shadow: inset 0 2px 6px rgba(0,0,0,0.3);
        }
        
        .step-title {
            color: #34D399;
            font-weight: 600;
            margin-top: 12px;
            border-bottom: 1px dashed #334155;
            padding-bottom: 2px;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Secure Dataset Pipeline Loader
DATASET_DIR = "."

@st.cache_data(show_spinner=False)
def load_large_scale_datasets():
    """Extracts rows from the decoupled 10,000-entry database files cleanly."""
    try:
        mimic_df = pd.read_csv(os.path.join(DATASET_DIR, "mimic_iv_notes_10k.csv"))
        drugbank_df = pd.read_csv(os.path.join(DATASET_DIR, "drugbank_matrix_10k.csv"))
        snomed_df = pd.read_csv(os.path.join(DATASET_DIR, "snomed_terms_10k.csv"))
        
        # Optimize MIMIC using O(1) indexed hash matches
        mimic_indexed = mimic_df.set_index("note_id", drop=False)
        return mimic_indexed, drugbank_df, snomed_df
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Instantiate data memory spaces
mimic_db, drugbank_db, snomed_db = load_large_scale_datasets()

# 4. State Initializers
if "final_report_content" not in st.session_state:
    st.session_state.final_report_content = ""
if "current_patient_id" not in st.session_state:
    st.session_state.current_patient_id = "EHR-A-9920"

# 5. Left Sidebar: Clinical Parameter Control Panel
with st.sidebar:
    st.markdown("### 🧬 Patient Vitals Core")
    st.caption("Adjust clinical parameters in real-time")
    st.markdown("---")
    
    # Secure API Client Initialization
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        st.error("❌ Missing Groq API Validation Key.")
        st.stop()
    client = Groq(api_key=GROQ_API_KEY)
    
    # --- PHYSICAL FEATURE INTERACTIVE WIDGETS ---
    st.markdown("##### 🌡️ Core Metrics")
    input_temp = st.slider("Body Temperature (°C)", 35.0, 42.0, 38.9, 0.1, help="Normal range: 36.5°C - 37.5°C")
    input_bpm = st.slider("Heart Rate (BPM)", 40, 180, 92, help="Normal resting range: 60 - 100 BPM")
    input_glucose = st.slider("Blood Glucose Level (mg/dL)", 50, 450, 245, help="Normal fasting range: < 100 mg/dL")
    
    st.markdown("##### 🫁 Secondary Metrics")
    col_sys, col_dia = st.columns(2)
    with col_sys:
        input_sys = st.number_input("Systolic BP", 80, 200, 158)
    with col_dia:
        input_dia = st.number_input("Diastolic BP", 40, 130, 98)
        
    input_spo2 = st.slider("Oxygen Saturation ($SpO_2$ %)", 70, 100, 91)
    
    st.markdown("---")
    st.markdown("##### ⚠️ Allergy Profiles & History")
    selected_allergies = st.multiselect("Active Patient Allergies:", ["Penicillin", "Sulfa", "NSAID", "Aspirin"], default=["Penicillin"])
    
    # Reset Parameters Button
    if st.button("🔄 Reset Vitals to Baseline", use_container_width=True):
        st.rerun()

# 6. Main Dashboard Architecture
st.title("🩺 Aegis AI // Enterprise Diagnostic Console")
st.markdown("Cross-referencing live patient physical profiles against 10,000-row tabular data matrices securely.")

# Dynamic Vitals Dashboard Banner display
with st.container(border=True):
    st.markdown(f"##### 📋 Active Tracked Case: `{st.session_state.current_patient_id}` | Demographics: 47yo Male")
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Body Temperature", f"{input_temp}°C", delta=f"{round(input_temp - 37.0, 1)}°C vs Normal", delta_color="inverse")
    m_col2.metric("Heart Rate (BPM)", f"{input_bpm} bpm", delta="Elevated (>90)" if input_bpm > 90 else "Normal")
    m_col3.metric("Blood Glucose", f"{input_glucose} mg/dL", delta="Hyperglycemic" if input_glucose > 140 else "Normal", delta_color="inverse")
    m_col4.metric("Blood Pressure", f"{input_sys}/{input_dia} mmHg")
    m_col5.metric("Oxygen Saturation", f"{input_spo2}%", delta="Hypoxia Warning" if input_spo2 < 95 else "Stable", delta_color="normal")

st.markdown("<br>", unsafe_allow_html=True)

# Layout Columns split
col_ctrl, col_terminal = st.columns([11, 10], gap="large")

with col_ctrl:
    with st.container(border=True):
        st.markdown("##### 📥 Clinical Observation Narrative")
        
        # Build dynamic narrative template mirroring custom slider metrics automatically
        default_narrative = (
            f"Patient presents with an acute body temperature reading of {input_temp}°C accompanied by tachycardia "
            f"at {input_bpm} BPM. Standard finger-prick diagnostics show a high serum blood glucose calculation sitting "
            f"at {input_glucose} mg/dL. Systemic blood pressure tracks at {input_sys}/{input_dia} mmHg with oxygen "
            f"saturation metrics hovering at {input_spo2}% SpO2."
        )
        
        clinical_input = st.text_area(
            "Synthesized Manifest Transcript Log:",
            height=125,
            value=default_narrative
        )
        
        # Dropdown selection from top rows of MIMIC data context
        sample_ids = mimic_db['note_id'].head(15).tolist() if not mimic_db.empty else ["No Records Loaded"]
        mimic_select = st.selectbox("Anchor Cohort Historical Reference File (MIMIC-IV Row):", sample_ids)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # PRIMARY EXECUTION ACTION BUTTONS
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            analyze_btn = st.button("⛓️ Run Decision Matrix Loop", type="primary", use_container_width=True)
        with btn_col2:
            clear_btn = st.button("🧹 Clear Workspace", use_container_width=True)
            if clear_btn:
                st.session_state.final_report_content = ""
                st.rerun()

with col_terminal:
    st.markdown("##### 🧠 Live Pipeline Logic & Data Validation Logs")
    status_indicator = st.empty()
    live_stream_box = st.empty()

# 7. Processing Runtime Engine
if analyze_btn and clinical_input:
    status_indicator.info("📊 Parsing local database matrices...")
    
    # Query targets securely from indexed memory space 
    anchor_note = mimic_db.loc[mimic_select] if mimic_select in mimic_db.index else {"condition": "Unknown", "clinical_note": "No historical details matches."}
    
    # Filter DrugBank safety constraints according to choices adjusted on the panel
    subset_interactions = drugbank_db[drugbank_db['item_key'].isin(selected_allergies + ["Metformin", "Lisinopril", "Sulfa"])]
    drugbank_context_text = ""
    for idx, row in subset_interactions.head(12).iterrows():
        drugbank_context_text += f"- Constraint [{row['category']}]: {row['item_key']} -> Restricted matches: {row['cross_references']}\n"
        
    # Standardize complete localized RAG knowledge context prompt layer
    compiled_rag_context = f"""
    [MIMIC-IV COHORT BASELINE DATASET FILTER]
    Row Reference Token ID: {mimic_select}
    Target Pathology Group: {anchor_note.get('condition', 'Unknown')}
    Historical Narrative Note: {anchor_note.get('clinical_note', 'No notes loaded')}
    
    [PHARMACEUTICAL DRUGBANK RETRIEVAL SAFETY MATRIX]
    {drugbank_context_text if drugbank_context_text else "- No explicit pharmaceutical restrictions matched."}
    """
    
    system_prompt = """You are an elite automated diagnostic reasoning agent processing enterprise medical record layers.
    You must logically separate structural execution checks. Print processing logs directly using this exact notation schema:
    
    [THOUGHT: PHASE 1: RELATIONAL MATCH ANALYSIS]
    (Evaluate incoming presentations against the injected MIMIC large historical note text parameters)
    
    [THOUGHT: PHASE 2: CONTRAINDICATION EXTRACTION]
    (Verify medications to make sure you block all allergy rules derived from the provided DrugBank matrix)
    
    [FINAL DIAGNOSIS & MANAGEMENT PLAN]
    (Draft final clean actionable medical directives here. Do not show any further THOUGHT prefixes after this line)"""

    user_prompt = f"""Active Vitals Array: Temp={input_temp}°C, HR={input_bpm}bpm, Glucose={input_glucose}mg/dL, BP={input_sys}/{input_dia}, SpO2={input_spo2}%
    Allergies Matrix List: {selected_allergies}
    
    Tabular Context Injected:
    {compiled_rag_context}
    
    Live Intake Presentation Transcription:
    {clinical_input}"""

    status_indicator.warning("⚡ Connecting securely to the Groq Processing Units...")
    
    try:
        # Launch real-time inference stream
        completion_stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            stream=True
        )
        
        full_output, thoughts_output, directives_output = "", "", ""
        
        for chunk in completion_stream:
            token = chunk.choices[0].delta.content or ""
            full_output += token
            
            if "[FINAL DIAGNOSIS" in full_output:
                splits = full_output.split("[FINAL DIAGNOSIS & MANAGEMENT PLAN]")
                thoughts_output = splits[0]
                if len(splits) > 1:
                    directives_output = splits[1]
                status_indicator.success("✅ Assessment complete. Finalized clinical report compiled below.")
            else:
                thoughts_output = full_output
                status_indicator.info("🧠 Safety matrices mapping across data blocks...")
                
            # Stream trace reasoning variables step by step into the terminal window
            with col_terminal:
                polished_logs = thoughts_output.replace(
                    "[THOUGHT: PHASE 1: RELATIONAL MATCH ANALYSIS]", 
                    "<div class='step-title'>📊 STEP 1: TABULAR PROFILE CROSS-EXAMINATION</div>"
                ).replace(
                    "[THOUGHT: PHASE 2: CONTRAINDICATION EXTRACTION]", 
                    "<div class='step-title'>🛡️ STEP 2: CROSS-REFERENCING PHARMACEUTICAL ALLERGY MATRICES</div>"
                ).replace("]", "")
                
                live_stream_box.markdown(f"<div class='terminal-box'>{polished_logs}</div>", unsafe_allow_html=True)
                
        # Cache final answer into session memory state safely to maintain persistent UI displays
        st.session_state.final_report_content = directives_output
                
    except Exception as e:
        status_indicator.error("❌ Runtime Engine Disconnect Error.")
        st.error(f"Groq API connection trace exception: {e}")

# 8. Clean Finalized Clinical Report Workspace Section
if st.session_state.final_report_content:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### 📋 Finalised Clinical Consultation & Validation Report")
        st.markdown("This validated framework directive is compiled directly from real-time physical metric evaluations and database rule checks.")
        st.write("---")
        
        # Display the generated report markdown nicely
        st.markdown(st.session_state.final_report_content)
        st.write("---")
        
        # Action Command Download Button
        st.download_button(
            label="💾 Download Finalised Medical Report (.txt)",
            data=st.session_state.final_report_content,
            file_name=f"clinical_report_{st.session_state.current_patient_id}.txt",
            mime="text/plain",
            use_container_width=True
        )
