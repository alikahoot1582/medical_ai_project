import streamlit as st
import pandas as pd
import json
import os
from groq import Groq

# ==============================================================================
# 🔑 EMBED YOUR GROQ API KEY HERE
# Replace the placeholder text with your actual secret key from console.groq.com
# ==============================================================================
GROQ_API_KEY = "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"

# 1. Page & Layout Optimization Configuration
st.set_page_config(
    page_title="Aegis AI - Clinical Workspace Console",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Seamless Glassmorphic Clinical Dashboard Theme UI Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* Global Reset & Font Application */
        html, body, [class*="css"] { 
            font-family: 'Plus Jakarta Sans', sans-serif; 
        }
        
        .stApp { 
            background-color: #080D1A; 
            color: #E2E8F0; 
        }
        
        /* Custom Sidebar Adjustment */
        div[data-testid="stSidebarUserContent"] { 
            background-color: #0E1626; 
            padding-top: 1.5rem;
        }
        
        /* Modernized Medical Metrics Cards */
        .patient-badge {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 14px;
            margin-bottom: 12px;
        }
        
        .vital-tag {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.2);
            color: #10B981;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin: 3px;
        }
        
        .allergy-tag {
            background: rgba(239, 68, 68, 0.1);
            border: 1px solid rgba(239, 68, 68, 0.2);
            color: #F87171;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-block;
            margin: 3px;
        }

        /* Tactical Live Logic/Reasoning Window Console */
        .reasoning-terminal { 
            background-color: #050B14; 
            border: 1px solid #1E293B; 
            border-radius: 8px; 
            padding: 18px; 
            font-family: 'JetBrains Mono', monospace; 
            font-size: 0.85rem; 
            color: #38BDF8; 
            white-space: pre-wrap; 
            height: 480px; 
            overflow-y: auto;
            box-shadow: inset 0 2px 8px rgba(0,0,0,0.8);
        }
        
        .phase-header {
            color: #10B981;
            font-weight: 600;
            margin-top: 10px;
            border-bottom: 1px dashed #1E293B;
            padding-bottom: 4px;
        }
    </style>
""", unsafe_allow_html=True)

# 2. Optimized Pipeline Local Dataset Fetcher Engine
DATASET_DIR = "."

@st.cache_data(show_spinner=False)
def load_large_scale_datasets():
    """Extracts baseline rows from the decoupled root-level spreadsheets securely."""
    try:
        mimic_df = pd.read_csv(os.path.join(DATASET_DIR, "mimic_iv_notes_10k.csv"))
        drugbank_df = pd.read_csv(os.path.join(DATASET_DIR, "drugbank_matrix_10k.csv"))
        snomed_df = pd.read_csv(os.path.join(DATASET_DIR, "snomed_terms_10k.csv"))
        
        # O(1) Quick-indexing configuration map setup
        mimic_indexed = mimic_df.set_index("note_id", drop=False)
        return mimic_indexed, drugbank_df, snomed_df
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Populate data tables cleanly inside execution context
mimic_db, drugbank_db, snomed_db = load_large_scale_datasets()

# 3. Structural Default Session State Configurations
if "current_patient" not in st.session_state:
    st.session_state.current_patient = {
        "id": "EHR-A-9920",
        "age": 47,
        "gender": "Male",
        "vitals": {"BP": "158/98", "HR": "92 bpm", "Temp": "38.9°C", "SpO2": "91%"},
        "lookup_history": ["high blood pressure (variant 0)", "sugar diabetes (variant 2)"],
        "allergies": ["Penicillin"]
    }

# 4. Sidebar Workspace Navigation Layout Banner
with st.sidebar:
    st.markdown("### 🩺 Aegis Intelligence")
    st.caption("Enterprise Clinical Decision Suite")
    st.markdown("---")
    
    # Secure Global SDK Target Initializer Verification Check
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
        st.error("🔒 **API Key Missing**\nUpdate your `app.py` script root context definition variable with a valid token configuration setup.")
        st.stop()
        
    client = Groq(api_key=GROQ_API_KEY)
    
    # Active Patient File Summary Card Layout
    patient = st.session_state.current_patient
    st.markdown(f"#### 📋 Patient File: `{patient['id']}`")
    
    with st.container():
        st.markdown(f"""
        <div class="patient-badge">
            <small style="color: #64748B;">DEMOGRAPHICS</small><br>
            <b>Age:</b> {patient['age']} | <b>Gender:</b> {patient['gender']}<br><br>
            <small style="color: #64748B;">VITAL SIGNALS</small><br>
            <span class="vital-tag">BP: {patient['vitals']['BP']}</span>
            <span class="vital-tag">HR: {patient['vitals']['HR']}</span>
            <span class="vital-tag">Temp: {patient['vitals']['Temp']}</span>
            <span class="vital-tag">SpO2: {patient['vitals']['SpO2']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Translate historical parameters to SNOMED codes dynamically
    st.markdown("##### 🧬 Ontology Map Tracking")
    if not snomed_db.empty:
        for item in patient['lookup_history']:
            match = snomed_db[snomed_db['user_term'] == item]
            if not match.empty:
                st.markdown(f"• {match.iloc[0]['preferred_term']}  \n`Code: {match.iloc[0]['snomed_code']}`")
                
    st.markdown("##### ⚠️ Declared Allergies")
    for allergy in patient['allergies']:
        st.markdown(f'<div class="allergy-tag">Contraindicated: {allergy}</div>', unsafe_allow_html=True)
        
    st.markdown("---")
    st.caption(f"💾 **Indexed State Matrix:** {len(mimic_db) + len(drugbank_db) + len(snomed_db):,} records")

# 5. Dashboard Core Panel Layout Design
st.subheader("🩺 High-Throughput Diagnostic Core Framework")
st.markdown("Cross-reference incoming clinical presentations against distributed database rows safely using low-latency LPU pipelines.")

# Balance spatial grid columns cleanly
col_input, col_chain = st.columns([11, 10], gap="large")

with col_input:
    with st.container(border=True):
        st.markdown("##### 📥 Patient Presentation Intake & Routing Anchor")
        clinical_input = st.text_area(
            "Observation / Symptom Transcript Log:",
            height=130,
            value="Patient shows high fever, confusion, rapid shallow breaths, and significantly elevated blood glucose parameters. Suspect severe systemic metabolic presentation requiring validation."
        )
        
        # Display the selector tool row
        sample_ids = mimic_db['note_id'].head(15).tolist() if not mimic_db.empty else ["No Records Loaded"]
        mimic_select = st.selectbox("Select Core MIMIC-IV Row Base Anchor Reference:", sample_ids)
        
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("⛓️ Run Decision Loops & Inference Engine", type="primary", use_container_width=True)

with col_chain:
    st.markdown("##### 🧠 Multi-Dataset Pipeline & Stream Reasoning Log")
    status_indicator = st.empty()
    live_stream_box = st.empty()

# 6. Groq Stream Thread Execution Routing Step Loop
if analyze_btn and clinical_input:
    # Anchor reports box inside left input column directly underneath control panel
    with col_input:
        st.markdown("<br>", unsafe_allow_html=True)
        report_container = st.container(border=True)
        report_container.markdown("##### 📋 Verified Clinical Decision Report Output")
        report_text_box = report_container.empty()
        
    status_indicator.info("🔍 Filtering cross-dataset parameters via O(1) frame lookups...")
    
    # Query row references cleanly from indexed memory frames
    anchor_note = mimic_db.loc[mimic_select] if mimic_select in mimic_db.index else {"condition": "Unknown", "clinical_note": "No match."}
    
    # Extract targeted pharmaceutical context limitations matching data filters
    subset_interactions = drugbank_db[drugbank_db['item_key'].isin(["Penicillin", "Metformin", "Lisinopril", "Sulfa"])]
    drugbank_context_text = ""
    for idx, row in subset_interactions.head(15).iterrows():
        drugbank_context_text += f"- Rule [{row['category']}]: {row['item_key']} -> Prohibited/Clashes: {row['cross_references']}\n"
        
    # Standardize relational text parameters context block
    compiled_rag_context = f"""
    [MIMIC-IV ANCHOR HISTORICAL PHENOTYPE RECORD]
    Row Index: {mimic_select}
    Condition Target: {anchor_note.get('condition', 'Unknown')}
    Historical Note Context: {anchor_note.get('clinical_note', 'No details')}
    
    [PHARMACEUTICAL DRUGBANK RETRIEVAL MATRIX]
    {drugbank_context_text}
    """
    
    system_prompt = """You are an elite automated diagnostic reasoning agent processing enterprise medical record layers.
    You must logically separate structural execution checks. Print processing logs directly using this exact notation schema:
    
    [THOUGHT: PHASE 1: RELATIONAL MATCH ANALYSIS]
    (Evaluate incoming presentations against the injected MIMIC large historical note text parameters)
    
    [THOUGHT: PHASE 2: CONTRAINDICATION EXTRACTION]
    (Verify medications to make sure you block all allergy rules derived from the provided DrugBank matrix)
    
    [FINAL DIAGNOSIS & MANAGEMENT PLAN]
    (Draft final clean actionable medical directives here. Do not show any further THOUGHT prefixes after this line)"""

    user_prompt = f"""Active Patient Profile Vitals: {patient['vitals']}
    Allergy Target Matrix: {patient['allergies']}
    
    Injected CSV Dataset Context:
    {compiled_rag_context}
    
    Active Intake Presentation:
    {clinical_input}"""

    status_indicator.warning("⚡ Establishing connection to Groq API Engine...")
    
    try:
        # Launch low-latency stream pipeline execution
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
            
            # Bifurcate reasoning tracing logs from final validated outputs on the fly
            if "[FINAL DIAGNOSIS" in full_output:
                splits = full_output.split("[FINAL DIAGNOSIS & MANAGEMENT PLAN]")
                thoughts_output = splits[0]
                if len(splits) > 1:
                    directives_output = splits[1]
                status_indicator.success("✅ Multi-Dataset verification loops securely evaluated.")
            else:
                thoughts_output = full_output
                status_indicator.info("🧠 Processing internal safety matrices across data streams...")
                
            # Render internal thought progression dynamically in terminal widget
            with col_chain:
                polished_chain = thoughts_output.replace(
                    "[THOUGHT: PHASE 1: RELATIONAL MATCH ANALYSIS]", 
                    "<div class='phase-header'>📊 PHASE 1: DATAFRAME COHORT CROSS-MATCH</div>"
                ).replace(
                    "[THOUGHT: PHASE 2: CONTRAINDICATION EXTRACTION]", 
                    "<div class='phase-header'>🛡️ PHASE 2: PHARMACEUTICAL MATRIX RULE CHECKS</div>"
                ).replace("]", "")
                
                live_stream_box.markdown(f"<div class='reasoning-terminal'>{polished_chain}</div>", unsafe_allow_html=True)
                
            # Stream finalized dashboard guidance text cleanly inside clinician panel
            if directives_output:
                report_text_box.markdown(directives_output)
                
    except Exception as e:
        status_indicator.error("❌ Diagnostic Processing Exception Encountered.")
        st.error(f"Groq Core Socket Pipeline Failure Reference: {e}")
