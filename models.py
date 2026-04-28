from google import genai

client = genai.Client(api_key="AIzaSyC4EIwUGvqLKej3J8gnjVE_Wa-tULU3x6c")

for m in client.models.list():
    print(m.name)