"""
The script will load the model and generate an answer based on the provided question and context

Set the script parameters:
--model_path: the path where the model is stored.
--question: the question you want the model to answer
--context: the relevant context for the question
--context_file: Path to a text file containing the context

Run the script:
Open a command line window
Navigate to the directory where the script is stored
Run the script using the following format:
    python script_name.py --model_path "path/to/model" --question "Your question" --context "Context relevant to the question"

Example:
    # Use direct context argument
    python command_line_test_model.py --model_path ".\checkpoint-12207\" --question "Who wrote Hamlet?" --context "Hamlet is a play written by a famous writer."
    # Or use context from file
    python command_line_test_model.py --model_path ".\checkpoint-12207\" --question "Who wrote Hamlet?" --context_file "test.txt"
"""

import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load model function
def load_model(model_path):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

# Function to get predictions based on the question and context
def get_prediction(question, context, model, tokenizer):
    input_text = f"question: {question} context: {context}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    outputs = model.generate(input_ids)
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return predicted_answer

# Main function
if __name__ == "__main__":
    # Setting Command Line Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="The path to the model directory.")
    parser.add_argument('--question', type=str, required=True, help="The question to ask the model.")
    parser.add_argument('--context', type=str, help="The context in which to ask the question.")
    parser.add_argument('--context_file', type=str, help="Path to a text file containing the context.")

    # Parse command line arguments
    args = parser.parse_args()

    # loading models and tokenizers
    model, tokenizer = load_model(args.model_path)

    # Determine context source
    if args.context_file:
        try:
            # Try reading with UTF-16 encoding
            with open(args.context_file, 'r', encoding='utf-16') as file:
                context = file.read()
        except UnicodeDecodeError:
            # If UTF-16 read fails, fall back to UTF-8 and then potentially other encodings
            with open(args.context_file, 'r', encoding='utf-8-sig') as file:
                context = file.read()
    else:
        # Use the context argument provided from the command line
        context = args.context

    # Validate that either context or context_file is provided
    if not context:
        raise ValueError("Either --context or --context_file must be provided.")

    # Get the prediction
    prediction = get_prediction(args.question, context, model, tokenizer)

    # Print predicted results
    print(f"Predicted results: {prediction}")

