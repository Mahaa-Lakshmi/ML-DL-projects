import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score,jaccard_score
from Levenshtein import distance as lev_distance
import warnings

warnings.filterwarnings("ignore")


# Sample ground truth
ground_truth = pd.read_csv("test_results/ground_truth.csv")  # filename, vendor, buyer, date, etc.
gemini_outputs = pd.read_csv("test_results/gemini_output_aggregated.csv")
mistral_outputs = pd.read_csv("test_results/mistral_output_aggregated.csv")

# Function to evaluate a single field
def field_accuracy(gt, pred, field):
    gt_series = gt[field].fillna("").astype(str).str.lower().str.strip()
    pred_series = pred[field].fillna("").astype(str).str.lower().str.strip()
    return accuracy_score(gt_series, pred_series)

#This measures how many edits (insertions, deletions, substitutions) are needed to turn one string into another.
def field_edit_distance(gt, pred, field):
    gt_series = gt[field].fillna("").astype(str).str.lower().str.strip()
    pred_series = pred[field].fillna("").astype(str).str.lower().str.strip()
    distances = [lev_distance(gt_i, pred_i) for gt_i, pred_i in zip(gt_series, pred_series)]
    return np.mean(distances)

def char_f1(gt, pred, field):
    gt_chars = "".join(gt[field].fillna("").astype(str).str.lower().str.strip())
    pred_chars = "".join(pred[field].fillna("").astype(str).str.lower().str.strip())
    gt_chars_bin = [c in pred_chars for c in gt_chars]
    pred_chars_bin = [c in gt_chars for c in pred_chars]
    
    # crude estimate; proper F1 needs sequence comparison (see SeqEval or TokenEval)
    precision = sum(pred_chars_bin) / len(pred_chars) if pred_chars else 0
    recall = sum(gt_chars_bin) / len(gt_chars) if gt_chars else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def jaccard_similarity(gt, pred, field):
    gt_series = gt[field].fillna("").astype(str).str.lower().str.split()
    pred_series = pred[field].fillna("").astype(str).str.lower().str.split()
    sims = []
    for g, p in zip(gt_series, pred_series):
        set_g, set_p = set(g), set(p)
        intersection = set_g.intersection(set_p)
        union = set_g.union(set_p)
        sims.append(len(intersection) / len(union) if union else 1)
    return np.mean(sims)


fields = ['seller', 'buyer', 'invoice_date', 'total', 'invoice_no','currency']

gemini_df,mistral_df=pd.DataFrame(columns=["field_accuracy","field_edit_distance","char_f1","jaccard_similarity"]),pd.DataFrame(columns=["field_accuracy","field_edit_distance","char_f1","jaccard_similarity"])

for field in fields:
    #gemini
    new_row_list = [field_accuracy(ground_truth, gemini_outputs, field), field_edit_distance(ground_truth, gemini_outputs, field), char_f1(ground_truth, gemini_outputs, field),jaccard_similarity(ground_truth, gemini_outputs, field)]
    new_row_df = pd.DataFrame([new_row_list], columns=gemini_df.columns,index=[field])
    gemini_df = pd.concat([gemini_df, new_row_df])

    #mistral
    new_row_list = [field_accuracy(ground_truth, mistral_outputs, field), field_edit_distance(ground_truth, mistral_outputs, field), char_f1(ground_truth, mistral_outputs, field),jaccard_similarity(ground_truth, mistral_outputs, field)]
    new_row_df = pd.DataFrame([new_row_list], columns=mistral_df.columns,index=[field])
    mistral_df = pd.concat([mistral_df, new_row_df])
    

print("============Gemini===================")
print(gemini_df)

print("============Mistral==================")
print(mistral_df)





