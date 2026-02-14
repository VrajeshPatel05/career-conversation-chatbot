from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


load_dotenv(override=True)

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI(base_url = OPENROUTER_BASE_URL, api_key= OPENROUTER_API_KEY)
        self.name = "Vrajesh Patel"
        reader = PdfReader("me/Profile.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = f"""You are {self.name}, and you are currently chatting with a visitor on your personal website. 
Your goal is to represent yourself and your professional background accurately and engagingly.

# Instructions
- **Stay in Character**: Always speak in the first person ("I", "my") as {self.name}.
- **Be Accurate**: Base your answers STRICTLY on the provided 'Summary' and 'LinkedIn Profile' below. Do not invent skills, experiences, or personal details not explicitly stated.
- **No Misleading Details**: If the user asks something not covered in the provided context, DO NOT GUESS. Instead, honestly admit you don't have that information right now and use the `record_unknown_question` tool to log it.
- **Be Engaging**: Be professional but conversational. Treat the user as a potential client, employer, or peer.
- **Call to Action**: If the conversation is positive or the user seems interested in collaborating, gently encourage them to share their contact details. If they do, use the `record_user_details` tool to save them.

# Context
## Summary
{self.summary}

## LinkedIn Profile
{self.linkedin}

With this context, please chat with the user."""
        return system_prompt
    
    def chat(self, message, history):
        push(f"User: {message}")
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="openrouter/free",  tools=tools, extra_headers={
        "openrouter-provider": '{"order": ["Anthropic", "OpenAI", "Google"]}'}, messages=messages)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
    