import os
from cerebras.cloud.sdk import Cerebras

client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

chat_history = []

with open('./SCA_Info.txt', 'r', encoding='utf-8') as file:
    file_contents = file.read()

context = {
    "role": "system", 
    "content": file_contents + "You are the patient's virtual assistant in a critical medical situation. ECG data indicates that the person is currently experiencing or at immediate risk of sudden cardiac arrest or death. Your task is to analyze the given facts,  inform the patient of their current health state for this condition, and provide immedidate step by step emergency instructions (calling emergency services, stopping phyiscal activity, taking medication as indicated by the patient's doctor, or alerting others) Use clear, calm, but urgent language. Be as concise as possible by removing filler words. The patient may respond with follow up questions."
}

chat_history.append(context)

def initial_user_notification():

    response = client.chat.completions.create(
        messages=chat_history,
        model="llama3.1-70b",
    )

    initial_response = response.choices[0].message.content

    chat_history.append({
        "role": "assistant",
        "content": initial_response
    })

    return initial_response

def talk_to_user(user_input):

    user_message = {"role": "user", "content": user_input}
    
    chat_history.append(user_message)

    response = client.chat.completions.create(
        messages=chat_history,
        model="llama3.1-70b",
    )
    
    assistant_response = response.choices[0].message.content

    chat_history.append({
        "role": "assistant",
        "content": assistant_response
    })
    
    return assistant_response

