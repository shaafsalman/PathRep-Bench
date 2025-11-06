import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from vllm import LLM, SamplingParams
from tabulate import tabulate
import json


class PathRepEvaluator:
    def __init__(self, model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9):
        """Initialize the evaluator with vLLM model."""
        print(f"Loading model: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=32768,
            disable_log_stats=True
        )
        # Increased max_tokens and enabled logprobs
        self.sampling_params = SamplingParams(
            temperature=0.001,
            max_tokens=4000,  # Increased from 500
            top_p=1.0,
            logprobs=1,
        )

        self.disease_list = [
            'Adrenocortical carcinoma',
            'Bladder Urothelial Carcinoma',
            'Breast invasive carcinoma',
            'Cholangiocarcinoma',
            'Colon adenocarcinoma',
            'Esophageal carcinoma',
            'Head and Neck squamous cell carcinoma',
            'Kidney Chromophobe',
            'Kidney renal clear cell carcinoma',
            'Kidney renal papillary cell carcinoma',
            'Liver hepatocellular carcinoma',
            'Lung adenocarcinoma',
            'Lung squamous cell carcinoma',
            'Mesothelioma',
            'Pancreatic adenocarcinoma',
            'Rectum adenocarcinoma',
            'Skin Cutaneous Melanoma',
            'Stomach adenocarcinoma',
            'Testicular Germ Cell Tumors',
            'Thyroid carcinoma',
            'Uveal Melanoma'
        ]

    def format_prompt(self, system_prompt, user_prompt):
        """Format prompt for the model - adjust as needed for your model's chat template."""
        prompt = f"<|system|>\n{system_prompt}</s>\n<|user|>\n{user_prompt}</s>\n<|assistant|>\n"
        return prompt

    def generate_response_single(self, prompt):
        """Generate a single response using vLLM and return the full generation object."""
        outputs = self.llm.generate([prompt], self.sampling_params)
        return outputs[0]

    def extract_score_from_gen(self, gen):
        """
        Robustly extract a numeric score (mean token logprob) from a vLLM generation object.
        Returns a float or None.
        """
        try:
            out = gen.outputs[0]
            
            # Try to get logprobs
            if hasattr(out, 'logprobs') and out.logprobs is not None:
                # logprobs is typically a list of dict objects
                vals = []
                for token_logprob_dict in out.logprobs:
                    if token_logprob_dict:
                        # Each dict has token_id as key and Logprob object as value
                        for logprob_obj in token_logprob_dict.values():
                            if hasattr(logprob_obj, 'logprob'):
                                vals.append(float(logprob_obj.logprob))
                            else:
                                # Sometimes it's just a float
                                try:
                                    vals.append(float(logprob_obj))
                                except:
                                    pass
                
                if vals:
                    return float(np.mean(vals))
            
            # Fallback: try cumulative_logprob
            if hasattr(out, 'cumulative_logprob') and out.cumulative_logprob is not None:
                return float(out.cumulative_logprob)
                
        except Exception as e:
            print(f"Warning: Could not extract score: {e}")
        
        return None

    def extract_json_answer(self, text, fallback_key='answer'):
        """
        Enhanced JSON extraction that handles incomplete responses.
        Returns the extracted value or 'no answer'.
        """
        try:
            # Remove think tags if present
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            
            # Try to find complete JSON first
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
                try:
                    extracted_dict = json.loads(json_str)
                    # Return first value found
                    for key in ['diagnosis', 'answer', 'stage', 'survival', 'prediction']:
                        if key in extracted_dict:
                            return str(extracted_dict[key]).strip()
                    # Return any first value
                    if extracted_dict:
                        return str(list(extracted_dict.values())[0]).strip()
                except json.JSONDecodeError:
                    # Try ast.literal_eval as fallback
                    try:
                        extracted_dict = ast.literal_eval(json_str)
                        if extracted_dict:
                            return str(list(extracted_dict.values())[0]).strip()
                    except:
                        pass
            
            # If JSON extraction fails, try to find answer in text
            # Look for common patterns
            patterns = [
                r'(?:answer|stage|diagnosis)[\s:]+([A-D]|Stage [IVX]+|True|False|[A-Z][a-z\s]+carcinoma)',
                r'\(([A-D])\)',
                r'(Stage [IVX]+)',
                r'(True|False)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
        except Exception as e:
            print(f"Warning: JSON extraction error: {e}")
        
        return "no answer"

    def print_full_log(self, question_num, task_name, system_prompt, user_prompt, 
                       full_prompt, raw_response, actual, predicted, score):
        """Print complete untruncated logs."""
        print("\n" + "="*100)
        print(f"{'='*40} QUESTION {question_num} - {task_name} {'='*40}")
        print("="*100)
        
        print("\n" + "-"*100)
        print("SYSTEM PROMPT:")
        print("-"*100)
        print(system_prompt)
        
        print("\n" + "-"*100)
        print("USER PROMPT:")
        print("-"*100)
        print(user_prompt)
        
        print("\n" + "-"*100)
        print("RAW MODEL RESPONSE (complete, no truncation):")
        print("-"*100)
        print(raw_response)
        
        print("\n" + "-"*100)
        print("EVALUATION:")
        print("-"*100)
        print(f"Actual Answer:    {actual}")
        print(f"Predicted Answer: {predicted}")
        print(f"Score (avg log probability): {score}")
        print(f"Match: {'✓ CORRECT' if str(predicted).lower().strip().replace(' ', '') == str(actual).lower().strip().replace(' ', '') else '✗ INCORRECT'}")
        print("-"*100)
        
        print("\n" + "="*100 + "\n")

    # ============= TASK 1: DISEASE CLASSIFICATION =============
    def run_task1_disease(self, test_csv, num_samples=None, output_dir="./results/task1"):
        print("\n" + "="*50)
        print("TASK 1: DISEASE CLASSIFICATION")
        print("="*50)

        test_df = pd.read_csv(test_csv)
        if num_samples:
            test_df = test_df.head(num_samples)

        reports = test_df['text'].tolist()
        true_labels = test_df['type_name'].tolist()

        system_prompt = """You are a highly knowledgeable pathology AI assistant. Extract the patient's diagnosis from the pathology report and output ONLY a JSON object with the diagnosis.

The diagnosis must be one of these options:
'Adrenocortical carcinoma', 'Bladder Urothelial Carcinoma', 'Brain Lower Grade Glioma', 'Breast invasive carcinoma', 'Cervical squamous cell carcinoma and endocervical adenocarcinoma', 'Cholangiocarcinoma', 'Colon adenocarcinoma', 'Esophageal carcinoma', 'Glioblastoma multiforme', 'Head and Neck squamous cell carcinoma', 'Kidney Chromophobe', 'Kidney renal clear cell carcinoma', 'Kidney renal papillary cell carcinoma', 'Liver hepatocellular carcinoma', 'Lung adenocarcinoma', 'Lung squamous cell carcinoma', 'Lymphoid Neoplasm Diffuse Large B-cell Lymphoma', 'Mesothelioma', 'Ovarian serous cystadenocarcinoma', 'Pancreatic adenocarcinoma', 'Pheochromocytoma and Paraganglioma', 'Prostate adenocarcinoma', 'Rectum adenocarcinoma', 'Sarcoma', 'Skin Cutaneous Melanoma', 'Stomach adenocarcinoma', 'Testicular Germ Cell Tumors', 'Thymoma', 'Thyroid carcinoma', 'Uterine Carcinosarcoma', 'Uterine Corpus Endometrial Carcinoma', 'Uveal Melanoma'

Output format: {"diagnosis": "exact diagnosis from list"}"""

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pred_file = f"{output_dir}/disease-predictions-{timestamp}.csv"
        details_file = f"{output_dir}/task1_details-{timestamp}.csv"

        preds = []
        correct = 0
        details_rows = []

        for i, (report, truth) in enumerate(zip(reports, true_labels)):
            user_prompt = f"What is the diagnosis? Output ONLY the JSON object.\n\nPathology Report:\n{report}"
            full_prompt = self.format_prompt(system_prompt, user_prompt)

            gen = self.generate_response_single(full_prompt)
            raw_text = gen.outputs[0].text if hasattr(gen.outputs[0], "text") else str(gen)
            score = self.extract_score_from_gen(gen)
            extracted_term = self.extract_json_answer(raw_text, 'diagnosis')

            preds.append(extracted_term)

            details_rows.append({
                "slno": i,
                "question": report,
                "actual": truth,
                "prediction": extracted_term,
                "raw_output": raw_text,
                "score": score,
                "full_prompt": full_prompt
            })

            # Print full untruncated log
            self.print_full_log(
                question_num=i+1,
                task_name="DISEASE CLASSIFICATION",
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                full_prompt=full_prompt,
                raw_response=raw_text,
                actual=truth,
                predicted=extracted_term,
                score=score
            )

            if extracted_term.lower().strip() == truth.lower().strip():
                correct += 1

        accuracy = correct / len(true_labels) if len(true_labels) > 0 else 0

        pd.DataFrame({"slno": range(len(preds)), "preds": preds}).to_csv(pred_file, index=False)
        pd.DataFrame(details_rows).to_csv(details_file, index=False)

        print(f"\n{'='*70}")
        print(f"TASK 1 SUMMARY")
        print(f"{'='*70}")
        print(f"Predictions saved to: {pred_file}")
        print(f"Details saved to: {details_file}")
        print(f"Task 1 Accuracy: {accuracy:.4f} ({correct}/{len(true_labels)})")
        print(f"{'='*70}\n")
        return accuracy, pred_file

    # ============= TASK 2: STAGE CLASSIFICATION =============
    def run_task2_stage(self, test_csv, num_samples=None, output_dir="./results/task2"):
        print("\n" + "="*50)
        print("TASK 2: STAGE CLASSIFICATION")
        print("="*50)

        test_df = pd.read_csv(test_csv)[['text', 'type_name', 'stage_overall']].dropna()
        if num_samples:
            test_df = test_df.head(num_samples)

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        details_file = f"{output_dir}/task2_details-{timestamp}.csv"
        all_accuracies = []
        details_rows = []

        for disease in self.disease_list:
            disease_df = test_df[test_df['type_name'] == disease].reset_index(drop=True)
            if len(disease_df) == 0:
                continue

            print(f"\nProcessing: {disease} ({len(disease_df)} samples)")
            correct = 0

            for i, row in disease_df.iterrows():
                system_prompt = """You are an expert pathology AI assistant. Identify the AJCC cancer stage from the pathology report.

Analyze the TNM classification in the report and determine the overall AJCC stage.

Output ONLY a JSON object in this exact format: {"answer": "Stage X"}
where X is one of: I, II, III, or IV"""

                user_prompt = f"""Identify the AJCC Stage from this pathology report.

Options:
(A) Stage I
(B) Stage II
(C) Stage III
(D) Stage IV

Pathology Report:
{row['text']}

Output format: {{"answer": "Stage X"}}"""

                prompt = self.format_prompt(system_prompt, user_prompt)
                gen = self.generate_response_single(prompt)
                raw_text = gen.outputs[0].text if hasattr(gen.outputs[0], "text") else str(gen)
                score = self.extract_score_from_gen(gen)
                extracted_stage = self.extract_json_answer(raw_text, 'answer')

                details_rows.append({
                    "disease": disease,
                    "slno": i,
                    "question": row['text'],
                    "actual": row['stage_overall'],
                    "prediction": extracted_stage,
                    "raw_output": raw_text,
                    "score": score,
                    "full_prompt": prompt
                })

                # Print full untruncated log
                self.print_full_log(
                    question_num=i+1,
                    task_name=f"STAGE CLASSIFICATION - {disease}",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    full_prompt=prompt,
                    raw_response=raw_text,
                    actual=row['stage_overall'],
                    predicted=extracted_stage,
                    score=score
                )

                # Normalize comparison
                actual_normalized = row['stage_overall'].lower().strip().replace(' ', '')
                pred_normalized = extracted_stage.lower().strip().replace(' ', '')
                
                if actual_normalized in pred_normalized or pred_normalized in actual_normalized:
                    correct += 1

            accuracy = correct / len(disease_df) if len(disease_df) > 0 else 0
            print(f"\n{disease} Accuracy: {accuracy:.4f} ({correct}/{len(disease_df)})")
            all_accuracies.append(accuracy)

        overall_accuracy = np.mean(all_accuracies) if all_accuracies else 0

        pd.DataFrame(details_rows).to_csv(details_file, index=False)
        
        print(f"\n{'='*70}")
        print(f"TASK 2 SUMMARY")
        print(f"{'='*70}")
        print(f"Details saved to: {details_file}")
        print(f"Task 2 Overall Accuracy: {overall_accuracy:.4f}")
        print(f"{'='*70}\n")
        return overall_accuracy, output_dir

    # ============= TASK 3: PROGNOSIS PREDICTION =============
    def run_task3_prognosis(self, test_csv, train_csv, val_csv, num_samples=None, output_dir="./results/task3"):
        print("\n" + "="*50)
        print("TASK 3: PROGNOSIS PREDICTION")
        print("="*50)

        test_df = pd.read_csv(test_csv)[['text', 'type_name', 'DSS.time']].dropna()
        test_df['DSS.time'] = test_df['DSS.time'].astype(float) / 365

        if num_samples:
            test_df = test_df.head(num_samples)

        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        full_df = pd.concat([train_df, val_df])
        full_df['DSS.time'] = full_df['DSS.time'].astype(float)

        disease_times = {
            disease: np.round(np.mean(full_df[full_df['type_name'] == disease]['DSS.time']) / 365, 2)
            for disease in self.disease_list
        }

        test_df['Survival_times'] = test_df['type_name'].map(disease_times)
        test_df['survival_over_mean'] = (test_df['DSS.time'] > test_df['Survival_times']).astype(str)

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        details_file = f"{output_dir}/task3_details-{timestamp}.csv"

        all_accuracies = []
        details_rows = []

        for disease in self.disease_list:
            disease_df = test_df[test_df['type_name'] == disease].reset_index(drop=True)
            if len(disease_df) == 0:
                continue

            print(f"\nProcessing: {disease} ({len(disease_df)} samples)")
            correct = 0

            for i, row in disease_df.iterrows():
                system_prompt = """You are an expert pathology AI assistant. Predict whether a patient will survive beyond a given time based on their pathology report.

Analyze the pathology features and output your prediction.

Output ONLY a JSON object in this exact format: {"answer": "True"} or {"answer": "False"}"""

                user_prompt = f"""Will the patient survive after {row['Survival_times']} years based on this pathology report?

Options:
(A) True - Patient will survive beyond {row['Survival_times']} years
(B) False - Patient will not survive beyond {row['Survival_times']} years

Pathology Report:
{row['text']}

Output format: {{"answer": "True"}} or {{"answer": "False"}}"""

                prompt = self.format_prompt(system_prompt, user_prompt)
                gen = self.generate_response_single(prompt)
                raw_text = gen.outputs[0].text if hasattr(gen.outputs[0], "text") else str(gen)
                score = self.extract_score_from_gen(gen)
                extracted_pred = self.extract_json_answer(raw_text, 'answer')

                details_rows.append({
                    "disease": disease,
                    "slno": i,
                    "question": row['text'],
                    "actual": row['survival_over_mean'],
                    "prediction": extracted_pred,
                    "raw_output": raw_text,
                    "score": score,
                    "full_prompt": prompt
                })

                # Print full untruncated log
                self.print_full_log(
                    question_num=i+1,
                    task_name=f"PROGNOSIS PREDICTION - {disease}",
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    full_prompt=prompt,
                    raw_response=raw_text,
                    actual=row['survival_over_mean'],
                    predicted=extracted_pred,
                    score=score
                )

                if extracted_pred.lower().strip() == row['survival_over_mean'].lower().strip():
                    correct += 1

            accuracy = correct / len(disease_df) if len(disease_df) > 0 else 0
            print(f"\n{disease} Accuracy: {accuracy:.4f} ({correct}/{len(disease_df)})")
            all_accuracies.append(accuracy)

        overall_accuracy = np.mean(all_accuracies) if all_accuracies else 0

        pd.DataFrame(details_rows).to_csv(details_file, index=False)
        
        print(f"\n{'='*70}")
        print(f"TASK 3 SUMMARY")
        print(f"{'='*70}")
        print(f"Details saved to: {details_file}")
        print(f"Task 3 Overall Accuracy: {overall_accuracy:.4f}")
        print(f"{'='*70}\n")
        return overall_accuracy, output_dir


def main():
    parser = argparse.ArgumentParser(description="Run PathRep-Bench evaluation on custom model")
    parser.add_argument("--model_path", type=str, default="rsjx/pathllama-3.1-8b", help="Path or HuggingFace model ID")
    parser.add_argument("--test_csv", type=str, default="./data/test.csv", help="Path to test.csv")
    parser.add_argument("--train_csv", type=str, default="./data/train.csv", help="Path to train.csv (for Task 3)")
    parser.add_argument("--val_csv", type=str, default="./data/val.csv", help="Path to val.csv (for Task 3)")
    parser.add_argument("--tasks", type=str, default="1,2,3", help="Tasks to run (comma-separated, e.g., '1,2,3')")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate per task (None = all)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization (0-1)")
    parser.add_argument("--output_dir", type=str, default="./results", help="Base output directory")
    args = parser.parse_args()

    evaluator = PathRepEvaluator(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization
    )

    tasks = [int(t.strip()) for t in args.tasks.split(",")]
    results = {}

    # Run requested tasks
    if 1 in tasks:
        acc, pred_dir = evaluator.run_task1_disease(
            args.test_csv,
            num_samples=args.num_samples,
            output_dir=f"{args.output_dir}/task1"
        )
        results["Task 1 - Disease Classification"] = f"{acc:.4f}"

    if 2 in tasks:
        acc, pred_dir = evaluator.run_task2_stage(
            args.test_csv,
            num_samples=args.num_samples,
            output_dir=f"{args.output_dir}/task2"
        )
        results["Task 2 - Stage Classification"] = f"{acc:.4f}"

    if 3 in tasks:
        acc, pred_dir = evaluator.run_task3_prognosis(
            args.test_csv,
            args.train_csv,
            args.val_csv,
            num_samples=args.num_samples,
            output_dir=f"{args.output_dir}/task3"
        )
        results["Task 3 - Prognosis Prediction"] = f"{acc:.4f}"

    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)

    table_data = [[task, acc] for task, acc in results.items()]
    print(tabulate(table_data, headers=["Task", "Accuracy"], tablefmt="grid"))

    results_file = f"{args.output_dir}/summary_results.json"
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
