from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()   # 환경변수 키 호출(.env)

client = OpenAI()

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