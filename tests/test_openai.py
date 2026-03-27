from openai import OpenAI
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

resp = client.responses.create(
    model=os.environ.get("LLM_MODEL", "gpt-5-mini"),
    input="Resume en 2 líneas qué es el artículo 69-B del CFF."
)

print(resp.output_text)
