# General Health Chatbot using Mistral-7B-Instruct

## Task Objective

The primary objective of this project was to develop a simple chatbot capable of answering general health-related questions. The chatbot is designed to act as a helpful and friendly medical assistant, providing informational responses while strictly avoiding the provision of specific medical advice, diagnoses, or treatment recommendations. A key aspect of the objective was to implement safety filters to prevent harmful or inappropriate responses.

## Dataset Used

No explicit external dataset was used for training or fine-tuning the Large Language Model (LLM) in this project. The chatbot leverages the pre-trained knowledge embedded within the `Mistral-7B-Instruct-v0.2` model. This model has been trained on a vast amount of text data, enabling it to understand and generate human-like text based on general knowledge. The "instruction-tuned" nature of the model means it has been further trained to follow instructions effectively.

## Models Applied

* **Large Language Model (LLM):** `mistralai/Mistral-7B-Instruct-v0.2`
    * This is an open-source, instruction-tuned model available on Hugging Face. It was chosen for its strong performance in conversational AI tasks and its availability for use in environments like Google Colab with memory-saving techniques (4-bit quantization).
* **Libraries:**
    * `transformers`: Used for loading and interacting with the Mistral model and its tokenizer.
    * `accelerate`: Used for efficient model loading and inference, especially for large models.
    * `bitsandbytes`: Enables loading the model in 4-bit precision, significantly reducing memory usage to run on consumer-grade GPUs (like those available in Google Colab's free tier).
    * `torch`: PyTorch library for tensor operations and deep learning model execution.

## Key Results and Findings

* **Functional Chatbot:** Successfully implemented an interactive command-line chatbot that processes user queries and generates responses using the Mistral LLM.
* **Effective Prompt Engineering:** The use of a clear system message (`"You are a helpful and friendly medical assistant..."`) effectively guided the LLM to adopt the desired persona and adhere to specified constraints. This resulted in responses that were generally informative, clear, and easy to understand.
* **Basic Safety Filter Implementation:** Rule-based safety filters were implemented to detect common phrases indicating requests for medical advice, diagnoses, or prescriptions. When such phrases are detected, the chatbot gracefully declines to provide direct medical advice and instead directs the user to consult a healthcare professional.
* **Mandatory Disclaimer:** A general medical disclaimer is appended to every response, reinforcing the non-medical advice nature of the chatbot and emphasizing the importance of professional medical consultation.
* **Resource Efficiency:** By loading the `Mistral-7B-Instruct-v0.2` model in 4-bit precision, the project demonstrated that powerful LLMs can be run on resource-constrained environments like Google Colab's free GPU tier, making LLM development more accessible.
* **Informative Responses:** For general health questions like "What causes a sore throat?" or "How do I treat a common cold?", the chatbot provided accurate and helpful information, reflecting the pre-trained knowledge of the LLM.
* **Limitations of Rule-Based Safety:** While functional, the rule-based safety filters are basic and can be circumvented. For robust, production-level medical chatbots, more sophisticated content moderation techniques (e.g., dedicated moderation APIs, fine-tuned safety models) would be necessary.
* **No Personal Medical Advice:** The project successfully adhered to its objective of not providing specific medical advice, instead consistently recommending professional medical consultation for personalized care.

This project serves as a strong foundational example for building LLM-powered conversational agents with a focus on defined roles and safety constraints.

---

## How to Run This Notebook

This project is implemented as a Google Colab notebook, making it easy to run directly in your browser.

1.  **Open in Colab:** Click the "Open in Colab" badge below to run the notebook directly in your browser:

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saleha321/General-Health-Query-Chatbot/blob/main/Health_Chatbot.ipynb)

2.  **Change Runtime Type:** Once the notebook opens in Colab, go to `Runtime` -> `Change runtime type` and select `T4 GPU` as the hardware accelerator.
3.  **Install Libraries:** Run the first code cell to install the necessary Python libraries (`transformers`, `accelerate`, `bitsandbytes`).
4.  **Hugging Face Access:** Since `Mistral-7B-Instruct-v0.2` is a gated model, you need to:
    * Accept the terms on the [Mistral-7B-Instruct-v0.2 Hugging Face page](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).
    * Generate a Hugging Face API token from your [Hugging Face settings](https://huggingface.co/settings/tokens) (with `read` role).
    * Run the `login()` cell in Colab and paste your token when prompted.
5.  **Load Model:** Run the cell that loads the tokenizer and the Mistral model. This step will download the model weights and may take several minutes.
6.  **Run Chatbot:** Execute the final cell to start the interactive chatbot. Type your questions and press Enter. Type `quit` to exit.

## Example Queries

* "What causes a sore throat?"
* "Is paracetamol safe for children?"
* "How do I treat a common cold?"
* "Can you diagnose my headache?" (This should trigger the safety filter)

## Disclaimer

**This chatbot provides general health information for educational purposes only and should not be considered medical advice. Always consult a qualified healthcare professional for any health concerns, diagnoses, or before making any decisions related to your health or treatment.**

## Author

saleha noor
https://github.com/Saleha321