import pandas as pd
import google.generativeai as genai
import re
import time
import random
from typing import List, Tuple, Dict
import os
from datetime import datetime

class GeminiMedicalQAEvaluator:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Gemini Medical QA Evaluator
        
        Args:
            api_key: Google AI API key for Gemini
            model_name: Name of the Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.results = []
        
    def load_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load the medical QA dataset from CSV
        
        Args:
            csv_path: Path to the CSV file containing questions and answers
            
        Returns:
            DataFrame with the loaded data
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded successfully: {len(df)} questions")
            print(f"Categories available: {df['Category'].unique()}")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def extract_answer_letter(self, answer_text: str) -> str:
        """
        Extract the correct answer letter from the answer text
        
        Args:
            answer_text: The full answer text from the dataset
            
        Returns:
            The letter of the correct answer (A, B, C, D, E, etc.)
        """
        if pd.isna(answer_text):
            return ""
        
        # Clean the answer text
        answer_text = str(answer_text).strip()
        
        # Look for patterns like "أ." or "ب." or "A." at the beginning
        patterns = [
            r'^([أ-ي])\.',  # Arabic letters with dot
            r'^([A-Z])\.',   # English letters with dot
            r'^([أ-ي])\s',   # Arabic letters with space
            r'^([A-Z])\s',   # English letters with space
        ]
        
        for pattern in patterns:
            match = re.search(pattern, answer_text)
            if match:
                letter = match.group(1)
                # Convert Arabic letters to English
                arabic_to_english = {
                    'أ': 'A', 'ب': 'B', 'ج': 'C', 'د': 'D', 'ه': 'E', 'هـ': 'E',
                    'و': 'F', 'ز': 'G', 'ح': 'H', 'ط': 'I', 'ي': 'J'
                }
                return arabic_to_english.get(letter, letter)
        
        # If no pattern found, return the first character if it's a letter
        if answer_text and answer_text[0].upper() in 'ABCDEFGHIJ':
            return answer_text[0].upper()
        
        return ""
    
    def create_few_shot_examples(self, df: pd.DataFrame, num_examples: int = 3) -> str:
        """
        Create few-shot examples from the dataset
        
        Args:
            df: DataFrame containing the questions
            num_examples: Number of examples to include
            
        Returns:
            Formatted few-shot examples string
        """
        # Select random examples from different categories
        categories = df['Category'].unique()
        examples = []
        
        for i in range(min(num_examples, len(categories))):
            category = categories[i]
            category_df = df[df['Category'] == category]
            if not category_df.empty:
                example = category_df.iloc[0]
                examples.append(example)
        
        few_shot_text = "Here are some examples of similar medical questions and their correct answers:\n\n"
        
        for i, example in enumerate(examples, 1):
            correct_answer = self.extract_answer_letter(example['Answer'])
            few_shot_text += f"Example {i}:\n"
            few_shot_text += f"Question: {example['Question']}\n"
            few_shot_text += f"Correct Answer: {correct_answer}\n\n"
        
        return few_shot_text
    
    def display_intermediate_results(self, category_results: Dict, total_predictions: int):
        """
        Display intermediate category results during evaluation
        
        Args:
            category_results: Current category results
            total_predictions: Total questions processed so far
        """
        if total_predictions % 10 == 0 and total_predictions > 0:  # Show every 10 questions
            print(f"\n--- Intermediate Results (after {total_predictions} questions) ---")
            for category, results in category_results.items():
                if results['total'] > 0:
                    acc = results['correct'] / results['total'] * 100
                    print(f"{category:15s}: {acc:5.1f}% ({results['correct']}/{results['total']})")
            print("-" * 50)
    
    def create_prompt(self, question: str, few_shot_examples: str) -> str:
        """
        Create the complete prompt with few-shot examples
        
        Args:
            question: The medical question to answer
            few_shot_examples: Few-shot examples string
            
        Returns:
            Complete formatted prompt
        """
        base_prompt = """You are given a medical multiple-choice question in Arabic, taken from real Arabic medical school exams and lecture notes. The question may come from any medical specialty (e.g., Histology, Pharmacology, Physiology) and may include complex clinical knowledge.

{few_shot_examples}

Now answer this question:

Question: {question}

Instructions:

Identify the full question and all provided answer choices (there may be more than four).

Consider any clinical, anatomical, or scientific context relevant to the question.

If visual references (graphs, tables) are mentioned, incorporate their implications in your decision-making.

Choose the correct answer based on the given options.

Reply with only the letter corresponding to the correct answer (e.g., A, B, C...). No explanation."""

        return base_prompt.format(few_shot_examples=few_shot_examples, question=question)
    
    def query_gemini(self, prompt: str, max_retries: int = 2) -> str:
        """
        Query the Gemini model with retry logic
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Maximum number of retries (default: 2)
            
        Returns:
            Model response or empty string if failed
        """
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # 2 second delay before retry
                else:
                    print(f"All {max_retries} attempts failed, skipping question")
                    return ""
        return ""
    
    def extract_predicted_answer(self, response: str) -> str:
        """
        Extract the predicted answer letter from Gemini's response
        
        Args:
            response: Raw response from Gemini
            
        Returns:
            Extracted answer letter
        """
        if not response:
            return ""
        
        response = response.strip()
        
        # Look for single letter responses
        if len(response) == 1 and response.upper() in 'ABCDEFGHIJ':
            return response.upper()
        
        # Look for patterns in the response
        patterns = [
            r'(?:answer|إجابة|الإجابة).*?([A-J])',  # "answer is A" or similar
            r'^([A-J])$',  # Just the letter
            r'^([A-J])\.',  # Letter with dot
            r'([A-J])(?:\s|$)',  # Letter followed by space or end
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If nothing found, return the first letter if it exists
        for char in response:
            if char.upper() in 'ABCDEFGHIJ':
                return char.upper()
        
        return ""
    
    def evaluate_dataset(self, df: pd.DataFrame, sample_size: int = None, few_shot_examples: int = 3) -> Dict:
        """
        Evaluate the model on the entire dataset or a sample
        
        Args:
            df: DataFrame containing questions and answers
            sample_size: Number of questions to evaluate (None for all)
            few_shot_examples: Number of few-shot examples to use
            
        Returns:
            Dictionary with evaluation results
        """
        # Sample the dataset if specified
        if sample_size and sample_size < len(df):
            df_sample = df.sample(n=sample_size, random_state=42)
            print(f"Evaluating on a sample of {sample_size} questions")
        else:
            df_sample = df.copy()
            print(f"Evaluating on all {len(df)} questions")
        
        # Create few-shot examples (excluding the test samples)
        few_shot_df = df[~df.index.isin(df_sample.index)]
        if few_shot_df.empty:
            few_shot_df = df.sample(n=min(few_shot_examples, len(df)), random_state=123)
        
        few_shot_text = self.create_few_shot_examples(few_shot_df, few_shot_examples)
        
        correct_predictions = 0
        total_predictions = 0
        category_results = {}
        
        print(f"Starting evaluation with {few_shot_examples} few-shot examples...")
        print("=" * 50)
        
        for idx, row in df_sample.iterrows():
            question = row['Question']
            true_answer = self.extract_answer_letter(row['Answer'])
            category = row['Category']
            
            if not true_answer:
                print(f"Skipping question {idx}: Could not extract true answer")
                continue
            
            # Create prompt with few-shot examples
            prompt = self.create_prompt(question, few_shot_text)
            
            # Query the model
            response = self.query_gemini(prompt)
            predicted_answer = self.extract_predicted_answer(response)
            
            # Skip question if API failed to respond
            if not response:
                print(f"⚠ Q{total_predictions + 1:3d}: {category:12s} | SKIPPED - API failed after retries")
                continue
            
            # Track results
            is_correct = predicted_answer == true_answer
            if is_correct:
                correct_predictions += 1
            total_predictions += 1
            
            # Track category results
            if category not in category_results:
                category_results[category] = {'correct': 0, 'total': 0}
            category_results[category]['total'] += 1
            if is_correct:
                category_results[category]['correct'] += 1
            
            # Store detailed results
            result = {
                'index': idx,
                'question': question[:100] + "..." if len(question) > 100 else question,
                'category': category,
                'true_answer': true_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'raw_response': response
            }
            self.results.append(result)
            
            # Print progress with category info
            accuracy = correct_predictions / total_predictions * 100
            status = "✓" if is_correct else "✗"
            category_acc = category_results[category]['correct'] / category_results[category]['total'] * 100
            print(f"{status} Q{total_predictions:3d}: {category:12s} | True: {true_answer} | Pred: {predicted_answer:1s} | Overall: {accuracy:5.1f}% | Cat: {category_acc:5.1f}%")
            
            # Show intermediate results every 10 questions
            self.display_intermediate_results(category_results, total_predictions)
            
            # Add 2 second delay between requests
            time.sleep(2)
        
        # Calculate final results
        overall_accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        
        # Calculate category accuracies
        for category in category_results:
            cat_correct = category_results[category]['correct']
            cat_total = category_results[category]['total']
            category_results[category]['accuracy'] = cat_correct / cat_total * 100 if cat_total > 0 else 0
        
        results_summary = {
            'overall_accuracy': overall_accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions,
            'category_results': category_results,
            'few_shot_examples': few_shot_examples
        }
        
        return results_summary
    
    def print_results(self, results: Dict):
        """
        Print evaluation results in a formatted way
        
        Args:
            results: Results dictionary from evaluate_dataset
        """
        print("\n" + "=" * 70)
        print("GEMINI MEDICAL QA EVALUATION RESULTS")
        print("=" * 70)
        
        # Overall Results
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   • Total Questions Evaluated: {results['total_predictions']}")
        print(f"   • Correct Predictions: {results['correct_predictions']}")
        print(f"   • Overall Accuracy: {results['overall_accuracy']:.2f}%")
        print(f"   • Few-shot Examples Used: {results['few_shot_examples']}")
        
        # Category-wise detailed results
        print(f"\n📋 CATEGORY-WISE PERFORMANCE:")
        print("-" * 70)
        print(f"{'Category':<20} {'Accuracy':<12} {'Correct':<8} {'Total':<8} {'Performance'}")
        print("-" * 70)
        
        # Sort categories by accuracy for better readability
        sorted_categories = sorted(results['category_results'].items(), 
                                 key=lambda x: x[1]['accuracy'], reverse=True)
        
        for category, cat_results in sorted_categories:
            accuracy = cat_results['accuracy']
            correct = cat_results['correct']
            total = cat_results['total']
            
            # Performance indicator
            if accuracy >= 80:
                performance = "🟢 Excellent"
            elif accuracy >= 70:
                performance = "🟡 Good"
            elif accuracy >= 60:
                performance = "🟠 Fair"
            else:
                performance = "🔴 Poor"
            
            print(f"{category:<20} {accuracy:>6.2f}%{'':<5} {correct:>3}/{total:<4} {performance}")
        
        print("-" * 70)
        
        # Summary statistics
        categories_count = len(results['category_results'])
        best_category = max(results['category_results'].items(), key=lambda x: x[1]['accuracy'])
        worst_category = min(results['category_results'].items(), key=lambda x: x[1]['accuracy'])
        avg_category_accuracy = sum(cat['accuracy'] for cat in results['category_results'].values()) / categories_count
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Categories Evaluated: {categories_count}")
        print(f"   • Average Category Accuracy: {avg_category_accuracy:.2f}%")
        print(f"   • Best Performing Category: {best_category[0]} ({best_category[1]['accuracy']:.2f}%)")
        print(f"   • Worst Performing Category: {worst_category[0]} ({worst_category[1]['accuracy']:.2f}%)")
        print(f"   • Performance Range: {worst_category[1]['accuracy']:.2f}% - {best_category[1]['accuracy']:.2f}%")
        
        print("=" * 70)
    
    def save_results(self, results: Dict, filename: str = None):
        """
        Save detailed results to CSV file
        
        Args:
            results: Results dictionary
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gemini_medical_qa_results_{timestamp}.csv"
        
        # Save detailed results
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(filename, index=False, encoding='utf-8-sig')
        
        # Save enhanced summary
        summary_filename = filename.replace('.csv', '_summary.txt')
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("GEMINI MEDICAL QA EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall results
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"Total Questions: {results['total_predictions']}\n")
            f.write(f"Correct Predictions: {results['correct_predictions']}\n")
            f.write(f"Overall Accuracy: {results['overall_accuracy']:.2f}%\n")
            f.write(f"Few-shot Examples: {results['few_shot_examples']}\n\n")
            
            # Category results
            f.write("CATEGORY-WISE RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'Category':<20} {'Accuracy':<10} {'Correct/Total'}\n")
            f.write("-" * 50 + "\n")
            
            # Sort categories by accuracy
            sorted_categories = sorted(results['category_results'].items(), 
                                     key=lambda x: x[1]['accuracy'], reverse=True)
            
            for category, cat_results in sorted_categories:
                accuracy = cat_results['accuracy']
                correct = cat_results['correct']
                total = cat_results['total']
                f.write(f"{category:<20} {accuracy:>6.2f}%     {correct:>2}/{total:<2}\n")
            
            f.write("-" * 50 + "\n\n")
            
            # Summary statistics
            categories_count = len(results['category_results'])
            best_category = max(results['category_results'].items(), key=lambda x: x[1]['accuracy'])
            worst_category = min(results['category_results'].items(), key=lambda x: x[1]['accuracy'])
            avg_category_accuracy = sum(cat['accuracy'] for cat in results['category_results'].values()) / categories_count
            
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"Categories Evaluated: {categories_count}\n")
            f.write(f"Average Category Accuracy: {avg_category_accuracy:.2f}%\n")
            f.write(f"Best Category: {best_category[0]} ({best_category[1]['accuracy']:.2f}%)\n")
            f.write(f"Worst Category: {worst_category[0]} ({worst_category[1]['accuracy']:.2f}%)\n")
            f.write(f"Performance Range: {worst_category[1]['accuracy']:.2f}% - {best_category[1]['accuracy']:.2f}%\n")
        
        # Create category-specific CSV
        category_summary_filename = filename.replace('.csv', '_category_summary.csv')
        category_data = []
        for category, cat_results in results['category_results'].items():
            category_data.append({
                'Category': category,
                'Accuracy_Percent': cat_results['accuracy'],
                'Correct_Answers': cat_results['correct'],
                'Total_Questions': cat_results['total'],
                'Questions_Wrong': cat_results['total'] - cat_results['correct']
            })
        
        df_category = pd.DataFrame(category_data)
        df_category = df_category.sort_values('Accuracy_Percent', ascending=False)
        df_category.to_csv(category_summary_filename, index=False, encoding='utf-8-sig')
        
        print(f"\nResults saved to:")
        print(f"  📄 Detailed results: {filename}")
        print(f"  📋 Summary report: {summary_filename}")
        print(f"  📊 Category summary: {category_summary_filename}")


def main():
    """
    Main function to run the evaluation
    """
    # Configuration
    API_KEY = "AIzaSyBpoe59ppZm2-c1j49ybDY3U4D_79QPZaU"  # Replace with your actual API key
    CSV_PATH = "multiple-choice-questions-old.csv"
    SAMPLE_SIZE = None  # Set to None to evaluate all questions, or specify a number for testing
    FEW_SHOT_EXAMPLES = 3
    
    # Initialize evaluator
    print("Initializing Gemini Medical QA Evaluator...")
    evaluator = GeminiMedicalQAEvaluator(api_key=API_KEY)
    
    # Load dataset
    print("Loading dataset...")
    df = evaluator.load_dataset(CSV_PATH)
    if df is None:
        return
    
    # Run evaluation
    print("Starting evaluation...")
    results = evaluator.evaluate_dataset(
        df=df, 
        sample_size=SAMPLE_SIZE, 
        few_shot_examples=FEW_SHOT_EXAMPLES
    )
    
    # Print and save results
    evaluator.print_results(results)
    evaluator.save_results(results)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
