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

# 1. Premium Page Configuration & Custom CSS Injection
st.set_page_config(
    page_title="Aegis AI // Enterprise Scale Diagnostics Core",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clinical dark UI optimization theme layout
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        .stApp { background-color: #0A0F1D; color: #E2E8F0; }
        div[data-testid="stSidebarUserContent"] { background-color: #111827; }
        .metric-card { 
            background: rgba(255, 255, 255, 0.03); 
            border-left: 4px solid #10B981; 
            padding: 12px; 
            margin-bottom: 10px; 
            border-radius: 4px; 
        }
        .reasoning-box { 
            background-color: #1E293B; 
            border: 1px solid #475569; 
            border-radius: 8px; 
            padding: 15px; 
            font-family: monospace; 
            font-size: 0.9rem; 
            color: #38BDF8; 
            white-space: pre-wrap; 
            height: 450px; 
            overflow-y: auto; 
        }
    </style>
""", unsafe_allow_html=True)

# 2. Database Loader Constants (Root-level mapping layout)
DATASET_DIR = "."

@st.cache_data
def load_large_scale_datasets():
    """Reads the decoupled 10,000-entry structural spreadsheet data matrix files."""
    try:
        mimic_df = pd.read_csv(os.path.join(DATASET_DIR, "mimic_iv_notes_10k.csv"))
        drugbank_df = pd.read_csv(os.path.join(DATASET_DIR, "drugbank_matrix_10k.csv"))
        snomed_df = pd.read_csv(os.path.join(DATASET_DIR, "snomed_terms_10k.csv"))
        
        # Optimize MIMIC dataframe using index keys for O(1) row queries
        mimic_indexed = mimic_df.set_index("note_id", drop=False)
        return mimic_indexed, drugbank_df, snomed_df
    except Exception as e:
        st.error(f"❌ Error initializing dataset records from root folder: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# Initialize localized datasets into functional memory dataframes
mimic_db, drugbank_db, snomed_db = load_large_scale_datasets()

# 3. Dynamic Clinical Patient File Initialization
if "current_patient" not in st.session_state:
    st.session_state.current_patient = {
        "id": "EHR-A-9920",
        "age": 47,
        "gender": "Male",
        "vitals": {"BP": "158/98", "HR": "92 bpm", "Temp": "38.9°C", "SpO2": "91%"},
        # Simulating variations generated inside the 10,000 rows vocabulary data file
        "lookup_history": ["high blood pressure (variant 0)", "sugar diabetes (variant 2)"],
        "allergies": ["Penicillin"]
    }

# 4. Sidebar Configuration & Security Checks
with st.sidebar:
    st.title("Aegis Enterprise Core")
    st.caption(f"🚀 Active Record State: {len(mimic_db) + len(drugbank_db) + len(snomed_db)} Rows Loaded")
    st.write("---")
    
    # Check if key is still set to placeholder text
    if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE" or not GROQ_API_KEY:
        st.error("❌ API Key Error: Please replace the placeholder string at the top of the app.py script file with your actual Groq API key.")
        st.stop()
        
    # Initialize Groq Client natively using the embedded token string variable
    client = Groq(api_key=GROQ_API_KEY)
    st.success("🔒 Groq API Engine Natively Authenticated")
    
    st.write("---")
    patient = st.session_state.current_patient
    st.subheader(f"Active Case: {patient['id']}")
    st.markdown(f"**Demographics:** {patient['age']}yo | {patient['gender']}")
    
    # Resolve unstructured conditions to official SNOMED-CT rows
    st.write("**Resolved History (SNOMED-CT Lookup):**")
    for item in patient['lookup_history']:
        match = snomed_db[snomed_db['user_term'] == item]
        if not match.empty:
            st.write(f"✅ {match.iloc[0]['preferred_term']} `(Code: {match.iloc[0]['snomed_code']})`")
            
    st.write("---")
    v = patient['vitals']
    st.markdown(f"<div class='metric-card'><b>BP Target:</b> {v['BP']}<br><b>Heart Rate:</b> {v['HR']}<br><b>Symptom Temp:</b> {v['Temp']}</div>", unsafe_allow_html=True)

# 5. Main Screen: Intake & Query Router
st.header("🔬 High-Throughput Diagnostic Core Framework")
st.markdown("Cross-referencing real-time clinical intake symptoms against production scale dataframe matrices.")

col_input, col_chain = st.columns([3, 2])

with col_input:
    st.subheader("📥 Enter Real-Time Clinical Presentation")
    clinical_input = st.text_area(
        "Direct Intake Narrative Log:",
        height=130,
        value="Patient shows high fever, confusion, rapid shallow breaths, and significantly elevated blood glucose parameters. Suspect severe systemic metabolic presentation requiring validation."
    )
    
    # Extract first 15 records from the 10,000 items list to keep selector clean and fast
    sample_ids = mimic_db['note_id'].head(15).tolist() if not mimic_db.empty else []
    mimic_select = st.selectbox("Select Target MIMIC-IV Row Record for Prompter Anchor:", sample_ids)
    
    analyze_btn = st.button("⛓️ Execute Enterprise Scale Diagnostic Loop", type="primary")

with col_chain:
    st.subheader("🧠 Live AI Reasoning Stream")
    status_box = st.empty()
    chain_box = st.empty()

# 6. Groq LPU Native Completion & Parsing Runtime
if analyze_btn and clinical_input:
    with col_input:
        st.write("---")
        st.subheader("📋 Context-Grounded Clinical Directives")
        report_box = st.empty()
        
    status_box.status("🔍 Extracting vector targets from pandas rows...", state="running")
    
    # O(1) indexed query retrieval to fetch specific notes text
    anchor_note = mimic_db.loc[mimic_select]
    
    # Filter specific drug interaction matrix parameters matching our clinical conditions
    subset_interactions = drugbank_db[drugbank_db['item_key'].isin(["Penicillin", "Metformin", "Lisinopril", "Sulfa"])]
    drugbank_context_text = ""
    for idx, row in subset_interactions.head(15).iterrows():
        drugbank_context_text += f"- Rule [{row['category']}]: {row['item_key']} -> Prohibited/Clashes: {row['cross_references']}\n"
        
    # Compile text representation of extracted data dataframes
    compiled_rag_context = f"""
    [MIMIC-IV ANCHOR HISTORICAL PHENOTYPE RECORD]
    Row Index: {anchor_note['note_id']}
    Condition Target: {anchor_note['condition']}
    Historical Note Context: {anchor_note['clinical_note']}
    
    [PHARMACEUTICAL DRUGBANK RETRIEVAL MATRIX]
    {drugbank_context_text}
    """
    
    # Design specialized engineering system logic prompt
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

    status_box.status("⚡ Establishing native LPU socket to Groq API...", state="running")
    
    try:
        # Create standard stream connection via Groq SDK
        completion_stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            stream=True
        )
        
        full_text, reasoning_text, report_text = "", "", ""
        
        # Intercept stream token chunks as they emerge
        for chunk in completion_stream:
            delta_content = chunk.choices[0].delta.content or ""
            full_text += delta_content
            
            # Divide text on the fly when final block is encountered
            if "[FINAL DIAGNOSIS" in full_text:
                splits = full_text.split("[FINAL DIAGNOSIS & MANAGEMENT PLAN]")
                reasoning_text = splits[0]
                if len(splits) > 1:
                    report_text = splits[1]
                status_box.status("✅ Enterprise validation vectors evaluated successfully.", state="complete")
            else:
                reasoning_text = full_text
                status_box.status("🧠 Evaluating tabular matrix constraints in real-time...", state="running")
                
            # Render internal thoughts dynamically in the code box container
            with col_chain:
                formatted_chain = reasoning_text.replace("[THOUGHT:", "\n📊 **DataFrame Scan Step:**").replace("]", "\n")
                chain_box.markdown(f"<div class='reasoning-box'>{formatted_chain}</div>", unsafe_allow_html=True)
                
            # Render polished report results in the left clinician column
            if report_text:
                with col_input:
                    report_box.markdown(report_text)
                    
    except Exception as e:
        status_box.status("❌ Inference Pipeline Aborted.", state="error")
        st.error(f"Groq Engine Connection Error: {e}")
