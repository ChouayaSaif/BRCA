import streamlit as st
import pandas as pd
import hashlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

# --------------------------
# ER STATUS PREDICTOR CLASS
# --------------------------
class ERStatusPredictor:
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._train_model()

    def _train_model(self):
        try:
            data = pd.read_csv('ER_1000_dna_sequences.csv')
            data['label'] = data['label'].apply(lambda x: 1 if x == 'ER_positive' else 0)
            X = data['dna_sequence']
            y = data['label']
            X_counts = self.vectorizer.fit_transform(X)
            self.model.fit(X_counts, y)
        except Exception as e:
            st.error(f"Error loading training data: {e}")

    def predict(self, dna_sequence: str) -> str:
        X = self.vectorizer.transform([dna_sequence])
        prediction = self.model.predict(X)[0]
        return 'Positive' if prediction == 1 else 'Negative'

# Cache the model to load it once per session
@st.cache_resource
def load_predictor():
    return ERStatusPredictor()

# --------------------------
# Digital Twin Pipeline Steps
# --------------------------
def analyze_genomics(tumor_dna: str, predictor) -> dict:
    er_status = predictor.predict(tumor_dna)
    mutations = {
        'PIK3CA': 'H1047R',
        'HER2': 'Amplified',
        'ER_status': er_status
    }
    return mutations

def predict_protein_from_rna(rna_seq: pd.DataFrame) -> dict:
    # Mock protein prediction - replace with real model as needed
    return {'HER2': 3.2, 'ER': 1.8}

def match_drugs(mutations: dict) -> list:
    drug_map = {
        'PIK3CA_H1047R': ['Alpelisib'],
        'HER2_Amplified': ['Trastuzumab', 'Pertuzumab'],
        'ER_Positive': ['Letrozole', 'Tamoxifen'],
        'ER_Negative': ['Chemotherapy']
    }
    drugs = []
    for gene, variant in mutations.items():
        key = f"{gene}_{variant}"
        drugs.extend(drug_map.get(key, []))
    return list(set(drugs))

def simulate_bioprinting(mutations: dict) -> str:
    er_status = mutations.get('ER_status', 'Unknown')
    return f"Bioprinting setup: material=Collagen I + Matrigel, cell_density=1e6 cells/mL, scaffold=BRCA model (ER {er_status})"

def log_to_blockchain(data: dict, private_key: str) -> str:
    data_str = str(data)
    tx_hash = hashlib.sha256(data_str.encode()).hexdigest()
    return f"Blockchain TX: {tx_hash} (Mock)"

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Digital Twin for Breast Cancer", layout="wide")
st.title("🧬 Digital Twin - Breast Cancer Simulation")

st.sidebar.header("Input Patient Data")
patient_id = st.sidebar.text_input("Patient ID", value="BRCA_001")
dna_sequence = st.sidebar.text_area("Tumor DNA Sequence", height=200)

if st.sidebar.button("Run Diagnosis"):
    if not dna_sequence or len(dna_sequence.strip()) < 50:
        st.warning("Please provide a valid DNA sequence with at least 50 characters.")
    else:
        with st.spinner("Running analysis..."):
            predictor = load_predictor()

            mutations = analyze_genomics(dna_sequence.strip(), predictor)
            protein_levels = predict_protein_from_rna(pd.DataFrame())
            drugs = match_drugs(mutations)
            bioprint_instructions = simulate_bioprinting(mutations)
            blockchain_tx = log_to_blockchain(
                {'mutations': mutations, 'drugs': drugs},
                private_key=f"patient_{patient_id}"
            )

        st.success("✅ Diagnosis Complete")

        st.subheader("🔬 Diagnosis Results")
        st.json(mutations)

        st.subheader("🧪 Predicted Protein Levels")
        st.json(protein_levels)

        st.subheader("💊 Recommended Drugs")
        if drugs:
            st.write(", ".join(drugs))
        else:
            st.write("No drug recommendations found for the given mutations.")

        st.subheader("🖨️ 3D Bioprinting Simulation")
        st.code(bioprint_instructions)

        st.subheader("⛓️ Blockchain Log")
        st.code(blockchain_tx)

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit")
