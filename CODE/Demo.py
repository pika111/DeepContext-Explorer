import logging
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd

# Load the model and tokenizer
model_path = "checkpoint-12207"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained(model_path)


def get_answer(question, context):
    try:
        if not question or not context:
            raise ValueError("Question and context cannot be empty.。")

        input_text = f"question: {question} context: {context}"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        outputs = model.generate(input_ids)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    except ValueError as e:
        return str(e)
    except Exception as e:
        # Log exception details, only a generic error message is returned here
        logging.error("An error occurred while processing the input.", exc_info=True)
        return "An error occurred while processing the request, please check your input and retry."


# Create Gradio interface
iface = gr.Interface(
    fn=get_answer,
    inputs=[
        gr.Textbox(lines=2, placeholder="Please enter a question...", label="question"),
        gr.Textbox(lines=5, placeholder="Please enter the context relevant to the question...", label="context")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Context-based Q&A System",
    description="You can enter a question and a relevant context (e.g. news) and the model will generate an answer to "
                "answer you based on the context."
                "<br><br>As an example: <br><strong> The relevant context:</strong>  \"Beijing is the capital of China, you can eat "
                "Peking duck and watch Peking Opera performances in Beijing\"."
                "<br><strong> Question:</strong> \"What performances can I watch in Beijing?\""
                "<br> <strong> Expected answer:</strong> \"Peking Opera performances\"",
    css=".gradio_container {font-family: Arial, sans-serif;} .gradio_container input, .gradio_container textarea, "
        ".gradio_container button {margin: 10px 0; width: 100%; padding: 10px;} .gradio_container button {"
        "background-color: #4CAF50; color: white; border: none; cursor: pointer;} .gradio_container button:hover {"
        "opacity: 0.8;}"
)

# Launch web interface
iface.launch(server_port=1000, share=True)

# Store test cases
results = []


def test_model(question, context, expected_answer):
    actual_answer = get_answer(question, context)
    result = {
        "question": question,
        "context": context,
        "expected_answer": expected_answer,
        "actual_answer": actual_answer,
        "status": "pass" if actual_answer == expected_answer else "fail"
    }
    results.append(result)
    return results


# Test cases
test_cases = [
    ("Who wrote Hamlet?", "Hamlet was written by Shakespeare.", "Shakespeare"),
    ("What is photosynthesis?",
     "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
     "A process used by plants to convert light energy into chemical energy."),
    ("Where is the Eiffel Tower located?", "The Eiffel Tower is located on the Champ de Mars in Paris, France.",
     "Paris, France"),
    ("What causes the seasons to change?",
     "The changing of seasons is caused by the tilt of the Earth's rotational axis away or toward the sun as it "
     "travels through its year-long path around the sun.",
     "The tilt of the Earth's rotational axis."),
    ("Who discovered penicillin?", "Penicillin was discovered by Alexander Fleming in 1928.", "Alexander Fleming"),
    ("How long is the Great Wall of China?", "The Great Wall of China is approximately 13,171 miles long.",
     "Approximately 13,171 miles"),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo.", "Tokyo"),
    ("What is the speed of light?", "The speed of light is exactly 299,792 kilometers per second.",
     "299,792 kilometers per second"),
    ("Who painted the Mona Lisa?", "The Mona Lisa was painted by Leonardo da Vinci.", "Leonardo da Vinci"),
    ("What is the tallest mountain in the world?",
     "Mount Everest is considered the tallest mountain in the world as measured from sea level.", "Mount Everest")
]

for question, context, expected in test_cases:
    res = test_model(question, context, expected)

# Save result to csv file
df = pd.DataFrame(results)
df.to_csv('test_results.csv', index=False)
