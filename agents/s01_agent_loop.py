#!/usr/bin/env python3
import json
import os
import subprocess

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

client = OpenAI()
MODEL = "gpt-5.4"
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."
TOOLS = [{
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]


def run_bash(command: str) -> str:
    if any(s in command for s in ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]):
        return "Error: Dangerous command blocked"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=os.getcwd(),
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:50000] if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


def agent_loop(messages: list):
    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM}] + messages,
            tools=TOOLS,
            max_completion_tokens=8000,
        )
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        assistant_message = {"role": "assistant", "content": message.content}
        if tool_calls:
            assistant_message["tool_calls"] = [tool_call.model_dump() for tool_call in tool_calls]
        messages.append(assistant_message)

        if not tool_calls:
            return

        for tool_call in tool_calls:
            if tool_call.function.name != "bash":
                output = f"Error: Unknown tool {tool_call.function.name}"
            else:
                args = json.loads(tool_call.function.arguments or "{}")
                command = args.get("command", "")
                print(f"[正在执行命令]: {command}")
                output = run_bash(command)
                print(output[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": output,
            })


if __name__ == "__main__":
    history = []
    while True:
        try:
            query = input("s01 >> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        response_content = history[-1]["content"]
        if response_content:
            print(response_content)
        print()
