from flask import Flask
from flask_pymongo import PyMongo
import csv, re
import random

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'aipla'
app.config['MONGO_URI'] = 'mongodb://firas:iamthegrandfox991@ds139251.mlab.com:39251/aipla'

mongo = PyMongo(app)

@app.route('/addall')
def addall():
    book = mongo.db.books
    with open('BOOKDATABASE.csv', 'r') as csvfile:
        books = csv.reader(csvfile, delimiter=',')
        for i in books:
            print(i[1])
            book.insert({'Title':i[1], 'Author' : i[2], 'Genre': i[3]})
    return 'Added book!'

@app.route('/find')
def find():
    book = mongo.db.books
    f = book.find_one({'Title':'Daughter of Sand and Stone'})
    print(f['Genre'])
    return "Done"

@app.route('/update')
def update():
    user = mongo.db.users
    Ali = user.find_one({'name': 'Ali'})
    Ali['language'] = 'Java'
    user.save(Ali)
    return 'Updated Ali!'

@app.route('/delete')
def delete():
    user = mongo.db.users
    fat = user.find_one({'name': 'Fatima'})
    user.remove(fat)
    return 'Removed!'


if __name__ == '__main__':
    app.run()