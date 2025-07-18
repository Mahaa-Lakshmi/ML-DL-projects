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
from llama_cpp import Llama
import torch

# Configure API key
genai.configure(api_key=os.getenv("GEMINI_KEY"))

# Initialize the model (e.g., Gemini Pro)
model = genai.GenerativeModel('gemini-2.0-flash')

MODEL_PATH = "models/gemma-2-2b-it-Q4_K_M.gguf"


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
        curr_date=date.today()

        if not input_data["invoice_date"] or not isinstance(input_data["invoice_date"], str) or input_data["invoice_date"].strip() == "":
            input_data["error_message"]="""Date not found. Please verify before submission."""

        elif datetime.strptime(input_data["invoice_date"], "%d/%m/%Y").date()>curr_date:
            #return r"""error": "Date incorrect. Please verify before submission."""
            input_data["error_message"]="""Date incorrect. Please verify before submission."""
        
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
    
#Gemma Agent for batch processing
class SemanticEntityAgentWithGemma(BaseAgent):
    def __init__(self):
        super().__init__("SemanticGemma")
        print("✅ Initialized SemanticGemma Agent")

        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=1024,
            n_threads=4,
            n_batch=128,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )

    def run(self, input_data: dict) -> dict:
        print("📥 Inside Gemma `run` method")
        raw_text = input_data.get("text", "")
        cleaned_text = "\n".join(line.strip() for line in raw_text.split("\n") if line.strip())

        prompt = self.build_prompt(cleaned_text)
        print("🚀 Sending prompt to Gemma...")

        try:
            response = self.llm(prompt, max_tokens=512, stop=["###"], echo=False)
            full_output = response["choices"][0]["text"].strip() if isinstance(response, dict) else response
            print("🧠 Raw Gemma Output:\n", full_output)
        except Exception as e:
            print("❌ LLM Execution Error:", e)
            return {"error": str(e)}

        parsed_json = self.extract_json(full_output)
        if parsed_json:
            parsed_json.update(input_data)
            return parsed_json
        else:
            return {"error": "Failed to parse JSON", "raw_output": full_output}

    def build_prompt(self, invoice_text: str) -> str:
        return f"""
            Extract the following fields from the invoice text and return them as a JSON object:
            - seller_name
            - invoice_no
            - invoice_date
            - buyer_name
            - total
            
            Format the invoice_date to "DD/MM/YYYY", and provide only the float for the total.
            
            Example Input:
            Lopez, Miller and Romero
            60464 Curtis Gateway
            East Keith, IN 57123
            
            Invoice Date: 05.08.2007
            Invoice No: 802205
            
            To: 
            Hayes LLC
            Mercedes Martinez
            960 Hurley Springs North
            Alyssa, RI 49322
            
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
            ###
            """

    def extract_json(self, text: str) -> dict | None:
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError) as e:
            print("⚠️ JSON parsing failed:", e)
            return None
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