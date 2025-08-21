from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_mysqldb import MySQL
import config

app = Flask(__name__)
CORS(app)

# Mysql
app.config['MYSQL_HOST'] = config.MYSQL_HOST
app.config['MYSQL_USER'] = config.MYSQL_USER
app.config['MYSQL_PASSWORD'] = config.MYSQL_PASSWORD
app.config['MYSQL_DB'] = config.MYSQL_DB
mysql = MySQL(app)
#home route
@app.route('/')
def home():
    return "Welcome API"

# get all users

@app.route('/students', methods=['GET'])
def get_students():
    currentDB = mysql.connection.cursor()
    currentDB.execute("SELECT * FROM students")
    data = currentDB.fetchall()
    currentDB.close()
    students = []
    for student in data:
        userData = {
            'id': student[0],
            'name': student[1],
            'class':student[2]
        }
        students.append(userData)
    return jsonify(students)



if __name__ == '__main__':
    app.run(debug=True)