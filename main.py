from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from io import StringIO
import traceback
import contextlib
import os

from dotenv import load_dotenv

# Google GenAI SDK
from openai import OpenAI
import json

# Load .env variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class CodeRequest(BaseModel):
    code: str

# Response model
class CodeResponse(BaseModel):
    error: List[int]
    result: str

# Structured AI output model
class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# -----------------------------
# STEP 1: Execute Python Code
# -----------------------------
def execute_python_code(code: str):
    """
    Execute Python code and return exact output.
    """

    output_buffer = StringIO()

    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code)

        output = output_buffer.getvalue()

        return {
            "success": True,
            "output": output
        }

    except Exception:
        output = traceback.format_exc()

        return {
            "success": False,
            "output": output
        }


# ---------------------------------
# STEP 2: AI Error Analysis
# ---------------------------------
def analyze_error_with_ai(code: str, traceback_text: str):

    client = OpenAI(
        api_key=os.getenv("AIPIPE_TOKEN"),
        base_url="https://aipipe.org/openrouter/v1"
    )

    prompt = f"""
You are a Python traceback analyzer.

Your task:
- Identify the exact line numbers where the error occurred.
- Use traceback carefully.
- Do not guess.
- Return ONLY valid JSON.

Example:
{{"error_lines":[3]}}

CODE:
{code}

TRACEBACK:
{traceback_text}
"""

    response = client.chat.completions.create(
        model="google/gemini-2.0-flash-lite-001",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    parsed = json.loads(content)

    return parsed["error_lines"]


# ---------------------------------
# STEP 3: API Endpoint
# ---------------------------------
@app.post("/code-interpreter", response_model=CodeResponse)
def code_interpreter(request: CodeRequest):

    execution_result = execute_python_code(request.code)

    # If success
    if execution_result["success"]:

        return {
            "error": [],
            "result": execution_result["output"]
        }

    # If error occurs
    error_lines = analyze_error_with_ai(
        request.code,
        execution_result["output"]
    )

    return {
        "error": error_lines,
        "result": execution_result["output"]
    }