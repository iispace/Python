""" 
ChatGPT as a Learning Tool

"""
# %%
from flask import Flask, session, request, render_template, jsonify
# from flask_session import Session

import openai, json, os, secrets 
import config


# %%
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

API_KEY = config.API_KEY

def chat_complete_request(system_persona, userText, myHyperParams):
    openai.api_key = API_KEY 

    if 'chat_history' not in session:
        session['chat_history']=[]

    message_item = {"role": "system", "content": f"You are a helpful {system_persona}."}
    session['chat_history'].append(message_item)
    
    message_item = {"role": "user", "content": f"{userText}"}
    session['chat_history'].append(message_item)

    parameters = { "model": "gpt-3.5-turbo", 
                "messages": session['chat_history'],    
            } 
    myHyperParams_json = json.loads(myHyperParams)

    parameters.update(myHyperParams_json)
    
    print(f"parameters: {parameters}")
    # Create chat completion 
    try:
        response = openai.ChatCompletion.create(**parameters)
    except Exception as e:
        return e
    
    # {"role": "assistant", "content": response.choices[0].message['content']}
    # n >1 인 경우 모든 응답을 messages에 저장 
    for resp in response.choices:
        message_item =  {"role": "assistant", "content": resp.message['content']}
        session['chat_history'].append(message_item)

    total_tokens = response['usage']['total_tokens']
    print(f"### Total_tokens in this chat session: {total_tokens} ###") 
    return response

@app.route("/")
def index():
    return render_template("index_1.html") # index.html 파일은 "templates" 폴더 하위에 있어야 함.

@app.route("/post", methods=["POST"])
def get_data_from_form():
    system_persona = request.json["system_persona"]
    userText = request.json["userInput"]
    myHyperParams = request.json["hyperparams"]
    
    chat_response = chat_complete_request(system_persona, userText, myHyperParams)

    if type(chat_response == openai.error.InvalidRequestError):
        print("======= OPENAI ERROR OCCURRED ===========")  
        err_msg = chat_response.args[0]
        err_code = 500
        print(f"error({err_code}): {err_msg}")
        return jsonify({'error': err_msg}), err_code
    
    return chat_response
    
if __name__ == "__main__":
    app.run(debug=True)
