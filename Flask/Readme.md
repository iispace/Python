## ChatGPT Prayground like UI implemented using Flask
<img src="https://github.com/iispace/Python/assets/24539773/f884205e-9a94-401e-aa82-98fbcb79d3c0" width=700>

## How to prepare coding environment using Flask in VSCode
  1. Create a virtual environment
     C:\project_folder>py -3 -m venv {myenvname}
  2. Activate the virtual environment
     C:\project_folder>.\\{myenvname}\Scripts\activate
  3. Install Flask, openai, and python-dotenv libraries in the virtual environment
     ({myenvname}) C:\project_folder>pip install Flask openai python-dotenv
  4. Create ".env" file in your project folder.
     C:\project_folder\.env
  5. Write the following in the ".env" file.
      FLASK_APP=app.py
      FLASK_ENV=development
      FLASK_RUN_PORT={any available port number you may want e.g. 8000}
  6. In your project folder, create a file with the name you assigned for "FLASK_APP" in ".env" file.
     In this case, the file name is to be "app.py"
  8. Open the "app.py" file and do your own coding.
  9. When you finish coding in the file, "app.py", run the application by following command:
     ({myenvname}) C:\project_folder>flask run
      
