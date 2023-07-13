import os
import openai

openai.api_key = os.environ("OPEN_AI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
          "role": "user",

          "content": """Generate a dataset for my NER model which contains LOC and PERSON entities.
                        The format should consist two elements: the first element is the text string, and the second element is a list of label annotations. Each label annotation is represented as a list containing three elements: the start index of the labeled entity the text, the end index of the labeled entity, and the label itself. the dataset must have list of arrays."
                        don't give any explainations.
                        """
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
print(response)

generated_data = response ["choices"][0]["message"]["content"].replace("\n","")

with open("data/clean/chatgpt_gen_train_data.txt", "w",encoding="utf-8") as f:
    f.write(generated_data)