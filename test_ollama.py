from ollama import chat

response = chat(
    model='qwen2.5:3b',
    messages=[
        {
            'role': 'user',
            'content': 'Say hello in one sentence.'
        }
    ]
)

print(response['message']['content'])