""" 
ToDO:

 2023-08-18:I used a global variable named "messages" but this is not good practice and not thread safe. The global variable needs to be replaced with session variable.
            The modification will be in progress soon.

"""
from flask import Flask, request, render_template, jsonify
import openai, json, os 
import config

app = Flask(__name__)

API_KEY = config.API_KEY
messages = []

def chat_complete_request(system_persona, userText, myHyperParams):
    openai.api_key = API_KEY 

    message_item = {"role": "system", "content": f"You are a helpful {system_persona}."}
    messages.append(message_item)
    message_item = {"role": "user", "content": f"{userText}"}
    messages.append(message_item)

    parameters = { "model": "gpt-3.5-turbo", 
                   "messages": messages,    
                } 
    myHyperParams_json = json.loads(myHyperParams)

    parameters.update(myHyperParams_json)
    
    print(f"parameters: {parameters}")
    # Create chat completion 
    response = openai.ChatCompletion.create(**parameters)
    message_item = {"role": "assistant", "content": response.choices[0].message['content']}
    messages.append(message_item)
    total_tokens = response['usage']['total_tokens']
    print(f"### Total_tokens in this chat session: {total_tokens} ###") 
    return response

@app.route("/")
def index():
    return render_template("index.html") # index.html should be located under "templates" folder 

@app.route("/post", methods=["POST"])
def get_data_from_form():
    system_persona = request.json["system_persona"]
    userText = request.json["userInput"]
    myHyperParams = request.json["hyperparams"]
    chat_response = chat_complete_request(system_persona, userText, myHyperParams)
     
    return chat_response
    
if __name__ == "__main__":
    app.run(debug=True)
