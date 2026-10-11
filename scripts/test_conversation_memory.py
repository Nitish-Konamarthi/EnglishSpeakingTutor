import requests

API_URL = "http://127.0.0.1:8080/v1"
MAX_HISTORY_MESSAGES = 10

SYSTEM_PROMPT = """
You are a friendly English-speaking tutor.

Your goals:
- Help the user practise natural English.
- Give concise, useful responses.
- Correct important English mistakes gently when appropriate.
- Ask relevant follow-up questions.
- Use earlier conversation details when answering follow-up questions.
- Never pretend to remember information that is not in the conversation.
"""


def get_model_id():
    """Retrieve the model identifier from the local llama.cpp server."""
    response = requests.get(f"{API_URL}/models", timeout=10)
    response.raise_for_status()

    models = response.json().get("data", [])

    if not models:
        raise RuntimeError("The local server did not return any available models.")

    return models[0]["id"]


def trim_history(history):
    """Keep recent conversation messages, removing complete oldest turns."""
    history = history.copy()

    while len(history) > MAX_HISTORY_MESSAGES:
        # Remove the oldest user message and its assistant response.
        if history and history[0]["role"] == "user":
            history.pop(0)

        if history and history[0]["role"] == "assistant":
            history.pop(0)

    return history


def get_tutor_reply(model_id, history, user_message):
    """Send conversation history and the new user message to local Qwen."""
    history.append({
        "role": "user",
        "content": user_message,
    })

    history[:] = trim_history(history)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
    ]

    response = requests.post(
        f"{API_URL}/chat/completions",
        json={
            "model": model_id,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 250,
        },
        timeout=180,
    )
    response.raise_for_status()

    reply = response.json()["choices"][0]["message"]["content"].strip()

    history.append({
        "role": "assistant",
        "content": reply,
    })

    return reply


def main():
    print("Connecting to the local Qwen server...")

    try:
        model_id = get_model_id()
    except requests.RequestException as exc:
        print(f"Could not connect to llama.cpp: {exc}")
        print("Make sure llama-server.exe is running on port 8080.")
        return

    print(f"Connected to model: {model_id}")

    # This list is the conversation's temporary memory.
    # It exists only while this Python process is running.
    history = []

    test_messages = [
        "My favourite language to learn is Spanish. I enjoy reading short stories in it.",
        "I find learning new vocabulary difficult. Can you suggest a practice method?",
        "Which language did I say is my favourite?",
    ]

    for turn_number, user_message in enumerate(test_messages, start=1):
        print(f"\n{'=' * 55}")
        print(f"TURN {turn_number}")
        print(f"You: {user_message}")

        try:
            reply = get_tutor_reply(model_id, history, user_message)
        except requests.RequestException as exc:
            print(f"Request failed: {exc}")
            return
        except (KeyError, IndexError, ValueError) as exc:
            print(f"Unexpected response from the model: {exc}")
            return

        print(f"Tutor: {reply}")
        print(f"Messages stored in memory: {len(history)}")

        if turn_number == 3:
            if "spanish" in reply.lower():
                print("\nMEMORY TEST: PASS")
                print("The tutor recalled the earlier language preference.")
            else:
                print("\nMEMORY TEST: NEEDS REVIEW")
                print("Check whether the tutor used the earlier conversation.")

    print("\nConversation-memory test completed.")


if __name__ == "__main__":
    main()
