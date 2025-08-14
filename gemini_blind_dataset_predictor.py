import pandas as pd
import google.generativeai as genai
import re
import time
import random
import os
import zipfile
from typing import List, Dict
from datetime import datetime

class GeminiBlindDatasetPredictor:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Gemini Blind Dataset Predictor
        
        Args:
            api_key: Google AI API key for Gemini
            model_name: Name of the Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.predictions = []
        
    def load_training_dataset(self, csv_path: str) -> pd.DataFrame:
        """
        Load the training medical QA dataset from CSV for few-shot examples
        
        Args:
            csv_path: Path to the training CSV file containing questions and answers
            
        Returns:
            DataFrame with the loaded training data
        """
        try:
            df = pd.read_csv(csv_path)
            print(f"Training dataset loaded successfully: {len(df)} questions")
            print(f"Categories available: {df['Category'].unique()}")
            return df
        except Exception as e:
            print(f"Error loading training dataset: {e}")
            return None
    
    def load_blind_dataset(self, tsv_path: str) -> List[str]:
        """
        Load the blind test dataset from TSV
        
        Args:
            tsv_path: Path to the TSV file containing blind test questions
            
        Returns:
            List of questions
        """
        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f.readlines()]
            print(f"Blind dataset loaded successfully: {len(questions)} questions")
            return questions
        except Exception as e:
            print(f"Error loading blind dataset: {e}")
            return []
    
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
    
    def create_few_shot_examples(self, training_df: pd.DataFrame, num_examples: int = 3) -> str:
        """
        Create few-shot examples from the training dataset
        
        Args:
            training_df: DataFrame containing the training questions
            num_examples: Number of examples to include
            
        Returns:
            Formatted few-shot examples string
        """
        # Select random examples from different categories if available
        if 'Category' in training_df.columns:
            categories = training_df['Category'].unique()
            examples = []
            
            for i in range(min(num_examples, len(categories))):
                category = categories[i]
                category_df = training_df[training_df['Category'] == category]
                if not category_df.empty:
                    example = category_df.iloc[0]
                    examples.append(example)
        else:
            # If no categories, just select random examples
            examples = training_df.sample(n=min(num_examples, len(training_df)), random_state=42)
        
        few_shot_text = "Here are some examples of similar medical questions and their correct answers:\n\n"
        
        for i, example in enumerate(examples, 1):
            if isinstance(example, pd.Series):
                question = example['Question']
                answer = example['Answer']
            else:
                question = example.iloc[0]  # Assuming first column is question
                answer = example.iloc[1]    # Assuming second column is answer
                
            correct_answer = self.extract_answer_letter(answer)
            few_shot_text += f"Example {i}:\n"
            few_shot_text += f"Question: {question}\n"
            few_shot_text += f"Correct Answer: {correct_answer}\n\n"
        
        return few_shot_text
    
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
                    print(f"All {max_retries} attempts failed, using fallback answer")
                    return ""
        return ""
    
    def extract_predicted_answer(self, response: str) -> str:
        """
        Extract the predicted answer letter from Gemini's response
        
        Args:
            response: Raw response from Gemini
            
        Returns:
            Extracted answer letter in Arabic format
        """
        if not response:
            # Fallback to random choice if API fails
            return random.choice(['أ', 'ب', 'ج', 'د', 'ه'])
        
        response = response.strip()
        
        # Look for single letter responses
        if len(response) == 1 and response.upper() in 'ABCDEFGHIJ':
            return self.english_to_arabic_letter(response.upper())
        
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
                return self.english_to_arabic_letter(match.group(1).upper())
        
        # If nothing found, return the first letter if it exists
        for char in response:
            if char.upper() in 'ABCDEFGHIJ':
                return self.english_to_arabic_letter(char.upper())
        
        # Final fallback to random choice
        return random.choice(['أ', 'ب', 'ج', 'د', 'ه'])
    
    def english_to_arabic_letter(self, english_letter: str) -> str:
        """
        Convert English letter to Arabic letter
        
        Args:
            english_letter: English letter (A, B, C, D, E)
            
        Returns:
            Arabic letter (أ, ب, ج, د, ه)
        """
        mapping = {
            'A': 'أ', 'B': 'ب', 'C': 'ج', 'D': 'د', 'E': 'ه',
            'F': 'و', 'G': 'ز', 'H': 'ح', 'I': 'ط', 'J': 'ي'
        }
        return mapping.get(english_letter, 'أ')  # Default to 'أ' if not found
    
    def predict_blind_dataset(self, 
                             blind_questions: List[str], 
                             training_df: pd.DataFrame, 
                             few_shot_examples: int = 3) -> List[str]:
        """
        Generate predictions for the blind dataset
        
        Args:
            blind_questions: List of blind test questions
            training_df: Training dataset for few-shot examples
            few_shot_examples: Number of few-shot examples to use
            
        Returns:
            List of predicted answers in Arabic letters
        """
        # Create few-shot examples from training data
        few_shot_text = self.create_few_shot_examples(training_df, few_shot_examples)
        
        predictions = []
        
        print(f"Starting prediction for {len(blind_questions)} questions...")
        print(f"Using {few_shot_examples} few-shot examples")
        print("=" * 60)
        
        for i, question in enumerate(blind_questions, 1):
            if not question.strip():
                continue
                
            # Create prompt with few-shot examples
            prompt = self.create_prompt(question, few_shot_text)
            
            # Query the model
            response = self.query_gemini(prompt)
            predicted_answer = self.extract_predicted_answer(response)
            
            predictions.append(predicted_answer)
            
            # Print progress
            print(f"Q{i:3d}: Predicted: {predicted_answer} | Response: {response[:50] if response else 'API_FAILED'}...")
            
            # Add 2 second delay between requests
            time.sleep(2)
        
        print("=" * 60)
        print(f"Prediction completed! Generated {len(predictions)} predictions")
        
        return predictions
    
    def save_predictions(self, 
                        predictions: List[str], 
                        filename: str, 
                        expected_count: int = 100):
        """
        Save predictions to CSV file in the required submission format
        
        Args:
            predictions: List of predicted answers
            filename: Output filename
            expected_count: Expected number of predictions (default: 100)
        """
        # Ensure we have exactly the expected number of predictions
        if len(predictions) < expected_count:
            # Pad with random answers if we have fewer predictions
            padding_needed = expected_count - len(predictions)
            random_answers = [random.choice(['أ', 'ب', 'ج', 'د', 'ه']) for _ in range(padding_needed)]
            predictions.extend(random_answers)
            print(f"⚠️  Padded {padding_needed} missing predictions with random answers")
        elif len(predictions) > expected_count:
            # Trim if we have too many predictions
            predictions = predictions[:expected_count]
            print(f"⚠️  Trimmed predictions to {expected_count} entries")
        
        # Save to CSV
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            for prediction in predictions:
                f.write(f"{prediction}\n")
        
        print(f"✅ Predictions saved to: {filename}")
        print(f"   Total predictions: {len(predictions)}")
    
    def create_submission_zip(self, prediction_files: List[str], zip_filename: str = None):
        """
        Create a zip file with prediction files for submission
        
        Args:
            prediction_files: List of prediction CSV files
            zip_filename: Name of the zip file (auto-generated if None)
        """
        if zip_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"medical_qa_submission_{timestamp}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in prediction_files:
                if os.path.exists(file_path):
                    # Add file to zip at root level (not in a folder)
                    zipf.write(file_path, os.path.basename(file_path))
                    print(f"✅ Added {file_path} to zip")
                else:
                    print(f"⚠️  File not found: {file_path}")
        
        print(f"🎯 Submission zip created: {zip_filename}")
        return zip_filename


def main():
    """
    Main function to generate predictions for blind dataset
    """
    # Configuration
    API_KEY = "AIzaSyBpoe59ppZm2-c1j49ybDY3U4D_79QPZaU"  # Replace with your actual API key
    TRAINING_CSV_PATH = "multiple-choice-questions-old.csv"  # Training data for few-shot
    BLIND_TSV_PATH = "subtask1_questions.tsv"  # Blind test questions
    FEW_SHOT_EXAMPLES = 3
    
    # Initialize predictor
    print("Initializing Gemini Blind Dataset Predictor...")
    predictor = GeminiBlindDatasetPredictor(api_key=API_KEY)
    
    # Load training dataset for few-shot examples
    print("Loading training dataset...")
    training_df = predictor.load_training_dataset(TRAINING_CSV_PATH)
    if training_df is None:
        print("❌ Failed to load training dataset. Exiting.")
        return
    
    # Load blind test dataset
    print("Loading blind test dataset...")
    blind_questions = predictor.load_blind_dataset(BLIND_TSV_PATH)
    if not blind_questions:
        print("❌ Failed to load blind dataset. Exiting.")
        return
    
    # Generate predictions
    print("Generating predictions...")
    predictions = predictor.predict_blind_dataset(
        blind_questions=blind_questions,
        training_df=training_df,
        few_shot_examples=FEW_SHOT_EXAMPLES
    )
    
    # Save predictions in submission format
    print("Saving predictions...")
    
    # For the main test dataset (assuming this is for subtask1_test)
    predictor.save_predictions(
        predictions=predictions,
        filename="predictions_subtask1_test.csv",
        expected_count=100
    )
    
    # Create submission zip
    prediction_files = ["predictions_subtask1_test.csv"]
    zip_filename = predictor.create_submission_zip(prediction_files)
    
    print("\n🎉 Prediction generation completed successfully!")
    print(f"📁 Submission file: {zip_filename}")
    print("\n📋 Summary:")
    print(f"   • Training examples used: {FEW_SHOT_EXAMPLES}")
    print(f"   • Questions processed: {len(blind_questions)}")
    print(f"   • Predictions generated: {len(predictions)}")
    print(f"   • Output files: {', '.join(prediction_files)}")


if __name__ == "__main__":
    main()
