import os
import requests
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = (
    "You are an assistant for a deepfake detection research system. "
    "Explain classifier predictions carefully. Never claim a classifier proves "
    "authenticity. Use retrieved evidence for factual background claims. "
    "Distinguish model output from general knowledge."
)


def generate_explanation(prediction, retrieved):
    # Use only the top 3 retrieved documents to reduce LLM memory usage
    retrieved = retrieved[:3]

    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434"
    )

    model = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2:3b"
    )

    # Prepare retrieved evidence
    evidence = "\n\n".join(
        f"[Source: {r['source']}]\n{r['text']}"
        for r in retrieved
    )

    # Prompt sent to the LLM
    user_prompt = f"""
Prediction:
Label: {prediction['label']}
Fake probability: {prediction['fake_probability']:.4f}
Real probability: {prediction['real_probability']:.4f}
Confidence: {prediction['confidence']:.4f}

Retrieved evidence:
{evidence}

Write a concise technical explanation containing:

1. Prediction summary.
2. Meaning of the confidence score.
3. Relevant information from the retrieved evidence.
4. Limitations of the prediction and why it is not proof of authenticity.

Keep the explanation short and easy to understand.
Do not invent evidence that is not provided above.
"""

    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "stream": False,
                "options": {
                    # Reduced context to prevent RAM/CPU memory errors
                    "num_ctx": 512,

                    # Lower temperature for more consistent explanations
                    "temperature": 0.2
                }
            },
            timeout=180
        )

        response.raise_for_status()

        result = response.json()

        return result["message"]["content"]

    except Exception as exc:
        return (
            "RAG retrieval succeeded, but the optional LLM explanation "
            f"is unavailable. Reason: {exc}"
        )