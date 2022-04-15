import mysql.connector

def database_connection():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="admin",
        database="forex"
    )
    return mydb