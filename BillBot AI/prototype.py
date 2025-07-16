from agents import OCRAgent,SemanticEntityAgentWithGemini,ValidationAgent,VisualizerAgent,SQLiteAgent
import streamlit as st
import sqlite3
import pandas as pd
import os
from PIL import Image
import tempfile


# Title & Sidebar
st.set_page_config(page_title="BillBot AI", layout="wide")
st.title("📄 BillBot AI – Invoice Analyzer")
st.sidebar.markdown("### Upload or Select Mode")
mode = st.sidebar.radio("Choose input mode:", ["Single Invoice", "Batch Processing"])

# Initialize agents
ocr_agent = OCRAgent()
entity_agent = SemanticEntityAgentWithGemini()
validator_agent = ValidationAgent()
visualizer_agent = VisualizerAgent()
sqlite_agent = SQLiteAgent("invoices.db")

# ---- SINGLE INVOICE MODE ----
if mode == "Single Invoice":
    uploaded_file = st.file_uploader("Upload an invoice image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            image_path = tmp_file.name

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_path, caption="📥 Uploaded Invoice", use_container_width=True)

            # === Run pipeline ===
            ocr_output = ocr_agent.run(image_path)
            entity_output = entity_agent.run(ocr_output)
            entity_output["filename"] = os.path.basename(uploaded_file.name)
            validation_output = validator_agent.run(entity_output)

            image, df = visualizer_agent.run({
                "image_path": image_path,
                "entities": validation_output["entities"]
            })

            # Insert into SQLite
            sqlite_agent.run(df)          

        
        with col2:
            st.image(image, caption="📌 OCR Bounding Box Output", use_container_width=True)

        # Display
        st.success("✅ Invoice processed successfully!") 
        for i in df.drop(columns=["text"]):
                   
        st.dataframe(df.drop(columns=["text"]), use_container_width=True)

# ---- BATCH MODE ----
elif mode == "Batch Processing":
    folder_path = st.text_input("Enter folder path to process batch invoices", value="backup/sampleDatasets")

    if st.button("🚀 Run Batch Pipeline"):
        if not os.path.exists(folder_path):
            st.error("❌ Folder not found.")
        else:
            all_dfs = []
            files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))]
            progress = st.progress(0)
            for i, file in enumerate(files):
                image_path = os.path.join(folder_path, file)
                try:
                    ocr_output = ocr_agent.run(image_path)
                    entity_output = entity_agent.run(ocr_output)
                    entity_output["filename"] = file
                    validation_output = validator_agent.run(entity_output)
                    df = pd.DataFrame([validation_output["entities"]])
                    sqlite_agent.run(df)
                    all_dfs.append(df)
                except Exception as e:
                    st.warning(f"⚠️ Failed to process {file}: {e}")
                progress.progress((i + 1) / len(files))

            if all_dfs:
                st.success(f"✅ Successfully processed {len(all_dfs)} invoices!")
                final_df = pd.concat(all_dfs, ignore_index=True)
                st.dataframe(final_df.drop(columns=["text"]), use_container_width=True)
                #download option
                csv = final_df.drop(columns=["text"]).to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name='invoices_output.csv',
                    mime='text/csv'
                )