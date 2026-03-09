import mysql.connector

def connect_db():

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="password",
        database="autoeda"
    )

    return conn