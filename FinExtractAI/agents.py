import os
import cv2
import pytesseract
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date,datetime
import json
import requests
import time
import google.generativeai as genai
import sqlite3
#from llama_cpp import Llama
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import InferenceClient,login

hugging_face_api_key=os.getenv("HUGGINGFACE_API")

login(hugging_face_api_key)

login(os.getenv("HUGGINGFACE_API"))

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_KEY"))

# Initialize the model (e.g., Gemini Pro)
model = genai.GenerativeModel('gemini-2.0-flash')

#MODEL_PATH = "models/gemma-2-2b-it-Q4_K_M.gguf"

pytesseract.pytesseract.tesseract_cmd = r"C:/Users/bmaha/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"


class BaseAgent:
    def __init__(self, name):
        self.name = name

    def run(self, input_data):
        raise NotImplementedError("Agent must implement run method")

    def __repr__(self):
        return f"{self.name} Agent"

#OCR AGENT

class OCRAgent(BaseAgent):
    def __init__(self):
        super().__init__("OCR")
        print("Inside OCR init")
        
    def cleaning_text(self,text):
        print("cleaning text OCR")
        text = re.sub(r'\n+', ' ', text)
        #text = re.sub(r'[^a-zA-Z0-9 ₹.,:/-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

    def run(self, image_path):
        print("ocr run")
        try:
            img = cv2.imread(image_path)
            filename=image_path.split("/")[-1]
            if img is None:
                raise ValueError(f"Image not loaded properly: {image_path}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            """blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)"""
            clean_text = pytesseract.image_to_string(gray)
            #print(clean_text)
            clean_text=self.cleaning_text(clean_text)
            return {"filename":filename,"text": clean_text}
        except Exception as e:
            print(f"[ERROR] Failed to process {image_path}: {e}")
            return ""

#SemanticExtraction Agent

#Custom semantic entity is not working properly due to noisy data so lets use gpt models

class SemanticEntityAgentWithGemini(BaseAgent):
    def __init__(self):
        super().__init__("SemanticEntity")
        print("Inside Gemini init")

    def run(self, input_data):
        print("Inside gemini run")
        text = input_data.get("text", "")
        text = [line.strip() for line in text.split("\n") if line.strip()]
        #curr_date = date.today()
        
        prompt = f"""
        Extract the following fields from the invoice text and return them as a JSON object:
        - seller_name
        - invoice_no
        - invoice_date
        - buyer_name
        - total

        Example Input:
        Lopez, Miller and Romero (Only  the name)
        60464 Curtis Gateway
        East Keith, IN 57123

        Invoice Date: 05.08.2007
        Invoice No: 802205

        To: 
        Hayes LLC (Only the name)
        Mercedes Martinez
        960 Hurley Springs North
        Alyssa, RI 49322

        Total: $534.11

        Example Output:
        
        Following things in JSON format of key:value
          "seller": "Lopez, Miller and Romero",
          "invoice_no": "802205",
          "invoice_date": "05/08/2007"(Put into this format "DD/MM/YYYY"),
          "buyer": "Hayes LLC",
          "currency" : "USD(3 Letter Currency code)"
          "total": "534.11 (only float values)"
   
        Now process this input:
        {text}
        """

        # Call the OpenAI API (you'll need to set up your API key first)
        response = model.generate_content(prompt)
        time.sleep(1)

        # ✅ Extract actual text from Gemini response
        output_text = response.candidates[0].content.parts[0].text.strip()

        # ✅ Remove markdown code block if present (like ```json ... ```)
        if output_text.startswith("```json"):
            output_text = output_text.strip("```json").strip("```").strip()

        #print("🔍 Gemini Output:\n", output_text)

        try:
            data = json.loads(output_text)
            #data["text"],data["filename"] = input_data.get("text", ""),input_data.get("filename", "")
            data.update(input_data)
            return data

        except json.JSONDecodeError:
            return {"error": "Failed to parse JSON", "raw_output": output_text}
        
#Validation Agent
        
class ValidationAgent(BaseAgent):
    def __init__(self):
        super().__init__("Validation")
        print("Inside validation init")

    def run(self, input_data):
        print("Inside validation run")
        missing_fields = []

        #validate input data
        #validate date
        parsed_invoice_date = None
        
        date_formats_to_try = [
            "%d/%m/%Y",  # DD/MM/YYYY (e.g., 15/07/2022)
            "%m/%d/%Y",  # MM/DD/YYYY (e.g., 12/25/2013)
            "%Y-%m-%d",  # YYYY-MM-DD (e.g., 2023-01-31)
            "%d-%m-%Y",  # DD-MM-YYYY (e.g., 31-01-2023)
            "%d.%m.%Y",  # DD.MM.YYYY (e.g., 05.08.2007) - from your example
            "%m.%d.%Y",  # MM.DD.YYYY (e.g., 08.05.2007)
            "%Y/%m/%d",  # YYYY/MM/DD (e.g., 2023/01/31)
        ]
        curr_date=date.today()

        if not input_data["invoice_date"] or not isinstance(input_data["invoice_date"], str) or input_data["invoice_date"].strip() == "":
            input_data["error_message"]="""Date not found. Please verify before submission."""

        else:
            date_str = input_data["invoice_date"].strip()
            for fmt in date_formats_to_try:
                try:
                    parsed_invoice_date = datetime.strptime(date_str, fmt).date()
                    # If parsing is successful, break the loop
                    break
                except ValueError:
                    continue # Try next format if current one fails
            
            if parsed_invoice_date is None:
                # If no format matched, set an error message
                input_data["error_message"] = f"Date format incorrect for '{date_str}'. Expected one of {', '.join(date_formats_to_try)}. Please verify before submission."
            elif parsed_invoice_date > curr_date:
                input_data["error_message"] = "Invoice date is in the future. Please verify before submission."
            # If parsing was successful and date is not in future, no error_message is set for date

        #extract amount and convert it to INR

        currency_code,amount=input_data["currency"],input_data["total"].strip()

        try:
            res = requests.get("https://v6.exchangerate-api.com/v6/f3f6bfc0330eb424583fd63b/latest/INR")
            rates = res.json()["conversion_rates"]
            rate = rates.get(currency_code, None)

            if not rate:
                #return {"error": f"Currency {currency_code} not found in exchange rate API."}
                input_data["Total_in_INR"]=amount
        
            else:
                # Step 5: Convert to INR
                converted_inr = round(float(amount) / rate, 2) if currency_code != "INR" else amount
                input_data["Total_in_INR"] = converted_inr

        except Exception as e:
            return {"error": "Exchange rate fetch failed", "details": str(e)}       


        for field in ["buyer", "seller", "invoice_date", "total"]:
            if field not in input_data:
                missing_fields.append(field)

        if missing_fields:
            input_data["error_message"] = "Following fields missing from image "+" ".join(map(str,missing_fields))

        print("===================Final input of Validation Agent============\n" , input_data)

        return {
            "status": "complete" if not missing_fields else "incomplete",
            "missing": missing_fields,
            "entities": input_data
        }


#Visualizer Agent

class VisualizerAgent(BaseAgent):
    def __init__(self):
        super().__init__("Visualizer")
        print("Inside visual init")

    def draw_boxes(self, image, data, column_text, color, col_name):
        print("Inside draw boxes")
        # Skip if the value is missing or empty
        if not column_text or not isinstance(column_text, str) or column_text.strip() == "":
            print(f"⚠️ Skipping box for '{col_name}' — value is empty or missing.")
            return image  # Just return the unchanged image

        target_words = column_text.lower().split()
        matches = []
        current_match = []

        for i, word in enumerate(data["text"]):
            word_lower = word.strip().lower()
            expected_word = target_words[len(current_match)] if current_match else target_words[0]

            if word_lower == expected_word:
                current_match.append(i)
                if len(current_match) == len(target_words):
                    matches.append(current_match.copy())
                    current_match = []
            else:
                current_match = []

        for match in matches:
            x_coords = [data["left"][i] for i in match]
            y_coords = [data["top"][i] for i in match]
            widths = [data["width"][i] for i in match]
            heights = [data["height"][i] for i in match]

            x_min = min(x_coords)-1
            y_min = min(y_coords)-1
            x_max = max([x_coords[i] + widths[i] for i in range(len(match))])+1
            y_max = max([y_coords[i] + heights[i] for i in range(len(match))])+1

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(image, col_name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    def run(self, input_data):
        print("Inside visual run")
        image_path = input_data["image_path"]
        extracted_entities = input_data["entities"]

        image = cv2.imread(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        color_map = {
            "seller": (9, 121, 105),
            "invoice_no": (255, 0, 0),
            "invoice_date": (0, 0, 255),
            "buyer": (0, 255, 255),
            "total": (255, 0, 255)
        }

        for col, color in color_map.items():
            if col in extracted_entities:
                self.draw_boxes(image, data, extracted_entities[col], color, col)

        # Display
        df = pd.DataFrame([extracted_entities])
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image_rgb,df


class SQLiteAgent(BaseAgent):
    def __init__(self, db_path):
        super().__init__("SQLite")
        print("Inside sqlite init")
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()

    def insert_invoice(self, df):
        try:
            required_columns = [
                "filename", "seller", "buyer", "invoice_no", "invoice_date",
                "currency", "total", "Total_in_INR", "error_message"
            ]

            # Add any missing columns with None (null)
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None

            # Reorder columns to match table
            df_sql = df[required_columns]

            # Insert to SQLite
            df_sql.to_sql("invoices", self.connection, if_exists="append", index=False)
            self.connection.commit()
            print("✅ Inserted DataFrame into SQLite")

        except Exception as e:
            print(f"[SQLiteAgent] ❌ Failed to insert invoice: {e}")

    def run(self, df):
        self.insert_invoice(df)

class SemanticEntityAgentWithMistral(BaseAgent):
    def __init__(self):
        super().__init__("SemanticMistral")
        print("✅ Initialized SemanticMistral Agent")

        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"

        client = InferenceClient(
            provider="featherless-ai",
            api_key=hugging_face_api_key,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

    def run(self, input_data: dict) -> dict:
        print("📥 Inside Mistral `run` method")
        raw_text = input_data.get("text", "")
        cleaned_text = "\n".join(line.strip() for line in raw_text.split("\n") if line.strip())

        prompt = self.build_prompt(cleaned_text)
        print("🚀 Prompting Mistral...")

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            print("🧠 Raw Mistral Output:\n", full_output)

            parsed_json = self.extract_json(full_output)
            if parsed_json:
                parsed_json.update(input_data)
                return parsed_json
            else:
                return {"error": "Failed to parse JSON", "raw_output": full_output}

        except Exception as e:
            print("❌ LLM Execution Error:", e)
            return {"error": str(e)}

    def build_prompt(self, invoice_text: str) -> str:
        return f"""
        Extract the following fields from the invoice text and return **only** a JSON object (no explanation, no formatting):
        - seller_name
        - invoice_no
        - invoice_date (format: DD/MM/YYYY)
        - buyer_name
        - total (float only)
        - currency (3-letter code)

        Example Input:
        Lopez, Miller and Romero
        Invoice Date: 05.08.2007
        Invoice No: 802205
        To: Hayes LLC
        Total: $534.11

        Expected Output:
        {{
          "seller": "Lopez, Miller and Romero",
          "invoice_no": "802205",
          "invoice_date": "05/08/2007",
          "buyer": "Hayes LLC",
          "currency": "USD",
          "total": "534.11"
        }}

        Now process this input:
        {invoice_text}
        """

    def extract_json(self, text: str) -> dict | None:
        try:
            # Try to find the last JSON object in the text
            json_matches = list(re.finditer(r"{[\s\S]*?}", text))
            if not json_matches:
                return None

            # Assume the last match is the actual output
            json_text = json_matches[-1].group()
            return json.loads(json_text)
        except Exception as e:
            print("⚠️ JSON parsing failed:", e)
            return None