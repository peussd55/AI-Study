from openai import OpenAI

OPENAI_API_KEY='YOUR-API-KEY'

client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model = 'gpt-4o-mini',
    messages=[{
        'role':'user',
        'temperature':0.0,
        'content': '한국의 수도는?'
    }]
)

print(completion)
print("===========================================================")
print(completion.choices[0].message.content)