import os
import openai

openai.api_key = os.environ("OPEN_AI_API_KEY")

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
          "role": "user",

          "content": """
            Find Location and Person entities in this text, just tell which words are the entities, not further explanation needed :
            At the Carmine Camelia, Kieran and Lauren end up being locked up in a supply closet after. To avoid drawing attention, they decide to wait until closing time to kick the door open.
            Lauren realizes that the failure in their spying devices happened because they both used the same type of technology, and the signals interfered with one another. She is uncomfortable by the whole situation and tells Kieran to stay away from her then sits further from him.
                        """
        }
    ],
    temperature=1,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
print(response["choices"][0]["message"]["content"])
