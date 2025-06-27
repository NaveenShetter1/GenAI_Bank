
# import libraries

import google.generativeai as genai

genai.configure(api_key='AIzaSyAscCPbKWrM41aBimhKexldkDt77XaeNk0')

# use a correct available free
model=genai.GenerativeModel('models/gemini-1.5-flash-8b')

# send input propmts

response=model.generate_content('Hi, what is the weather today in bangalore')
print(response.text)



# import google.generativeai as genai
# genai.configure(api_key="AIzaSyAscCPbKWrM41aBimhKexldkDt77XaeNk0")

# models = genai.list_models()
# for m in models:
#     print(m.name)
