import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from utils import JobInput

client = OpenAI(
    base_url='http://localhost:11434/v1/',

    # required but ignored
    api_key='ollama',
)

class OllamaEngine:
    def __init__(self):
        load_dotenv()
        print ("OllamaEngine initialized")

    async def generate(self, job_input):
        # Get model from OLLAMA_MODEL_NAME defauting to llama3.2:1b
        model = os.getenv("OLLAMA_MODEL_NAME", "llama3.2:1b")

        # Depending if prompt is a string or a list, we need to handle it differently and send it to the OpenAI API
        if isinstance(job_input.llm_input, str):
            if job_input.images:
                # Images require chat completions with multimodal content
                content = [{"type": "text", "text": job_input.llm_input}]
                for image_b64 in job_input.images:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    })
                openAiJob = JobInput({
                    "openai_route": "/v1/chat/completions",
                    "openai_input": {
                        "model": model,
                        "messages": [{"role": "user", "content": content}],
                        "stream": job_input.stream
                    }
                })
            else:
                # Buid new JobInput object with the OpenAI route and input
                openAiJob = JobInput({
                    "openai_route": "/v1/completions",
                    "openai_input": {
                        "model": model,
                        "prompt": job_input.llm_input,
                        "stream": job_input.stream
                    }
                })
        else:
            messages = job_input.llm_input
            # If images are provided, append them to the last user message
            if job_input.images:
                messages = [msg.copy() for msg in messages]
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i].get("role") == "user":
                        text = messages[i].get("content", "")
                        content = [{"type": "text", "text": text}] if isinstance(text, str) else text
                        for image_b64 in job_input.images:
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                            })
                        messages[i] = {**messages[i], "content": content}
                        break
            # Buid new JobInput object with the OpenAI route and input
            openAiJob = JobInput({
                "openai_route": "/v1/chat/completions",
                "openai_input": {
                    "model": model,
                    "messages": messages,
                    "stream": job_input.stream
                }
            })

        print ("Generating response for job_input:", job_input)
        print ("OpenAI job:", openAiJob)
        
        # Create a generator that will yield the response from the OpenAI API
        openAIEngine = OllamaOpenAiEngine()
        generate = openAIEngine.generate(openAiJob)

        # Yield the response from the OpenAI API
        async for batch in generate:
            yield batch

class OllamaOpenAiEngine(OllamaEngine):
    def __init__(self):
        load_dotenv()
        print ("OllamaOpenAiEngine initialized")

    async def generate(self, job_input):
        print("Generating response for job_input:", job_input)

        # Dump job_input to console
        openai_input = job_input.openai_input

        # for now e just mock the response
        if job_input.openai_route == "/v1/models":
            # Async response
            async for response in self._handle_model_request():
                yield response
        elif job_input.openai_route in ["/v1/chat/completions", "/v1/completions"]:
            async for response in self._handle_chat_or_completion_request(openai_input, chat=job_input.openai_route == "/v1/chat/completions"):
                yield response
        else:
            yield {"error": "Invalid route"}

    async def _handle_model_request(self):
        try:
            response = client.models.list()
            # build a json response from the response object
            # SyncPage[Model](data=[Model(id='llama3.2:1b', created=1737206544, object='model', owned_by='library')], object='list')\n
            yield {"object": "list", "data": [model.to_dict() for model in response.data]} 
        except Exception as e:
            yield {"error": str(e)}

    async def _handle_chat_or_completion_request(self, openai_input, chat=False):
        try:
            # Call openai.chat.completions.create or openai.completions.create based on the route
            if chat:
                response = client.chat.completions.create(**openai_input)
            else:
                response = client.completions.create(**openai_input)

            # If streaming is False, we can just return the response
            if not openai_input.get("stream", False):
                yield response.to_dict()
                return

            for chunk in response:
                # Log message to console
                print("Message:", chunk)
                # Return json of the chunk without any line breaks
                yield "data: " + json.dumps(chunk.to_dict(), separators=(',', ':')) + "\n\n"

            yield "data: [DONE]"
        except Exception as e:
            yield {"error": str(e)}