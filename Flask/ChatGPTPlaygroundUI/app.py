from flask import Flask, request, render_template, jsonify
import openai, json 
import config  

app = Flask(__name__)

# index.html의 jQuery에서 받은 리스트를 json 형식으로 변환하는 함수
def convert_to_dict(list_):
    dict = {}
    for item in list_:
        item = item.replace("{","").replace("}","").replace("\"","")
        k, v = item.split(":")
        value = float(v)
        if k == "n":
            value = int(v)
        dict[k] = value
    return dict   

@app.route("/")
def index():
    return render_template("index.html") # index.html 파일은 "templates" 폴더 하위에 있어야 함.

@app.route("/get")
def get_bot_response(): 
    # Define chat completion parameters
    openai.api_key = config.API_KEY 
  
    userTextwithParams = request.args.get("msg")
    userText = userTextwithParams.split(",")[0]
    myHyperParam=userTextwithParams.split(",")[1:] 
    myHyperParamDict = convert_to_dict(myHyperParam)
    
    # print(f"userText: {userText}")
    # print(f"myHyperParam: {myHyperParam}")
    # print(f"myHyperParamDict: {myHyperParamDict}")
    
    parameters = { "model": "gpt-3.5-turbo", 
                   "messages": [{"role": "system", "content": "You are a helpful assistant."}, 
                                {"role": "user", "content": f"{userText}"}], 
                    # "max_tokens": int(max_tokens), # 1024, 
                    # "temperature": float(temperature), #0.7, 
                    # "n": int(n), #1, 
                    # "stop": None, 
                    # "frequency_penalty": 0,
                    # "presence_penalty": 0
                } 
    parameters.update(myHyperParamDict)
    print(f"parameters: {parameters}")
    # Create chat completion 
    response = openai.ChatCompletion.create(**parameters) 
    answer = response.choices[0].message['content'] 
    return answer
    
if __name__ == "__main__":
    app.run()
