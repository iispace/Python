## How to prepare coding environment using Flask in VSCode
  1. Create a virtual environment<br>
     <code>C:\project_folder>py -3 -m venv {myenvname}</code>
  3. Activate the virtual environment<br>
     <code>C:\project_folder>.\\{myenvname}\Scripts\activate</code>
  4. Install Flask, openai, and python-dotenv libraries in the virtual environment<br>
     <code>({myenvname}) C:\project_folder>pip install Flask openai python-dotenv</code>
  5. Create ".env" file in your project folder.<br>
     <code>C:\project_folder\.env</code>
  6. Write the following in the ".env" file.<br>
      <pre>
      FLASK_APP=app.py
      FLASK_ENV=development
      FLASK_RUN_PORT={any available port number you may want e.g. 8000}
      FLASK_DEBUG=1</pre>
      <p><code>FLASK_DEBUG=1</code> is for activation of debug mode. Set this flag to <code>0</code> if you want to deactivate debug mode.</p>
  8. In your project folder, create a file with the name you assigned for "FLASK_APP" in ".env" file.
     In this case, the file name is to be "app.py"
  9. Open the "app.py" file and do your own coding.
  10. When you finish coding in the file, "app.py", run the application by following command:<br>
     <code>({myenvname}) C:\project_folder>flask run</code>
      
