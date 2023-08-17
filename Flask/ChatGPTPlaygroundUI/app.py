""" 
Ref: Python Flask Tutorial: Full-Featured Web App Part 1 - Getting Started (https://www.youtube.com/watch?v=MwZwr5Tvyxo)

"""
# %%
from flask import Flask, request, render_template, jsonify
import openai, json, os 
import config


# %%
app = Flask(__name__)

API_KEY = config.API_KEY

def convert_to_dict(list_):
    dict = {}
    for item in list_:
        item = item.replace("{","").replace("}","").replace("\"","")
        k, v = item.split(":")
        value = float(v)
        if k == "n" or k == "max_tokens":
            value = int(v)
        dict[k] = value
    return dict   

@app.route("/")
def index():
    return render_template("index_1.html") # index.html 파일은 "templates" 폴더 하위에 있어야 함.

@app.route("/get")
def get_bot_response(): 
    # Define chat completion parameters
    # openai.api_key = config.API_KEY 
    openai.api_key = API_KEY 
    
    contexts = request.args.get("msg").split(",")
    system_persona = contexts[0]
    userText = contexts[1]
    myHyperParam=contexts[2:] 
    myHyperParamDict = convert_to_dict(myHyperParam)
    
    parameters = { "model": "gpt-3.5-turbo", 
                   "messages": [{"role": "system", "content": f"You are a helpful {system_persona}."}, 
                                {"role": "user", "content": f"{userText}"}], 
                    # "stop": None, 
                    # "frequency_penalty": 0,
                    # "presence_penalty": 0
                } 
    parameters.update(myHyperParamDict)
    print(f"parameters: {parameters}")
    # Create chat completion 
    response = openai.ChatCompletion.create(**parameters) 
    return response
    
if __name__ == "__main__":
    app.run()
