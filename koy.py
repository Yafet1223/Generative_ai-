import google.generativeai as genai
import json

# CONFIG
genai.configure(api_key="AIzaSyCbnPOJhEIFCXKJkl0afGm9-EhX5JbryJ4")

model = genai.GenerativeModel("gemini-2.5-flash")

# TOOLS
def bible_search(query):
    return f"Search results for '{query}'"

def reflect(text):
    return f"Reflection: {text}"

def pray(text):
    return f"Prayer: {text}"

def respond(text):
    return f"Response: {text}"

def finish(text):
    return f"Final Answer: {text}"


# THINK
def think(goal, memory):
    prompt = f"""
You are an AI agent.

Goal: {goal}
Memory: {memory}

Return JSON:
{{
  "action": "scripture | reflect | pray | respond | finish",
  "input": "text"
}}
"""

    response = model.generate_content(prompt)
    text = response.text.strip()

    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        clean_json = text[json_start:json_end]
        return json.loads(clean_json)
    except:
        return {"action": "respond", "input": text}


# ACT
def act(action, input_text):
    if action == "scripture":
        return bible_search(input_text)
    elif action == "reflect":
        return reflect(input_text)
    elif action == "pray":
        return pray(input_text)
    elif action == "respond":
        return respond(input_text)
    elif action == "finish":
        return finish(input_text)
    else:
        return "Unknown action"


# AGENT LOOP
def agent(goal, max_steps=5):
    memory = []

    for step in range(max_steps):
        decision = think(goal, memory)

        action = decision.get("action")
        input_text = decision.get("input")

        result = act(action, input_text)

        memory.append(f"Step {step+1}: {action} -> {result}")

        if action == "finish":
            print(result)
            break


# RUN
if __name__ == "__main__":
    agent("Give me a bible verse about love and explain love")
