# 🧾 FinExtractAI — Intelligent Invoice Information Extractor

FinExtractAI is a multi-agent, modular pipeline for automated information extraction from invoice images. It combines OCR, LLM-based semantic parsing, and validation logic to extract structured fields (seller, buyer, invoice number, date, amount, currency) from unstructured documents.

---

## 💡 Features

- 🧠 **Multi-agent architecture** (modular, extensible design)
- 🖼️ **OCR preprocessing** using Tesseract
- 🔍 **Semantic entity extraction** with Gemini Flash & Mistral 7B
- ✅ **Validation agent** (date logic, currency conversion)
- 📦 **SQLite database logging** of extracted fields
- 📊 **Visualization agent** overlays fields on images
- 🔁 **A/B evaluation pipeline** for LLM performance


---

## 🛠️ Agents Overview

| Agent                              | Responsibility                                 |
|-------------------------------------|------------------------------------------------|
| `OCRAgent`                         | Extracts text from invoice images (Tesseract)  |
| `SemanticEntityAgentWithGemini`     | Structured fields via Gemini Flash               |
| `SemanticEntityAgentWithMistral`    | Structured fields via Mistral 7B               |
| `ValidationAgent`                   | Validates dates, converts currency to INR      |
| `VisualizerAgent`                   | Draws bounding boxes on fields                 |
| `SQLiteAgent`                       | Stores results in SQLite DB                    |

---

## 📁 Project Structure

```
.
├── agents.py                  # All core agents (OCR, LLMs, validation, etc.)
├── pipeline.py                # Pipeline orchestrating the agents
├── evaluate_ab.py             # Accuracy scoring for each field (Gemini vs Mistral)
├── data/
│   ├── ground_truth.csv
│   ├── gemini_output_aggregated.csv
│   └── mistral_output_aggregated.csv
├── visualizations/            # Saved annotated invoice images
├── README.md
└── requirements.txt
└── processing.ipynb            #Execution code Kaggle friendly

```

---

## 📈 Evaluation Metrics

- **Field-level accuracy:** Exact string match between predicted and ground truth values.
- **Other Evaluations:**  
  - Levenshtein distance (fuzzy string comparison) -  How many edits (insertions, deletions, substitutions) are needed to turn one string into another.  
  - F1 score evaluation  
  - Jaccard similarity

---

## 🧪 Sample Output

```json
{
  "seller": "Lopez, Miller and Romero",
  "invoice_no": "802205",
  "invoice_date": "05/08/2007",
  "buyer": "Hayes LLC",
  "currency": "USD",
  "total": "534.11",
  "Total_in_INR": "43925.00"
}
```

---

## 🔧 Setup & Usage

1. **Install requirements**
    ```sh
    pip install -r requirements.txt
    ```

2. **Set your API keys**
    - Google Gemini: `GEMINI_KEY`
    - Hugging Face: `HUGGINGFACE_API`

3. **Run the pipeline**
    ```sh
    streamlit run pipeline.py
    ```

4. **Evaluate accuracy**
    ```sh
    python evaluate_ab.py
    ```

---

## 🧠 What We’ll Learn / Demonstrate

- ✅ LLM orchestration for structured extraction
- ✅ Multi-agent design pattern for modular ML systems
- ✅ A/B evaluation of LLMs using field-level metrics
- ✅ OCR + NLP fusion for real-world document tasks
- ✅ Prompt engineering for few-shot examples
- ✅ Data validation & enrichment (e.g., currency conversion)
- ✅ Image annotation for debugging model predictions

---

**Happy extracting!**