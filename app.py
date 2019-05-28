import os, sys, re, csv
from flask import Flask, request
from flask_pymongo import PyMongo
from pprint import pprint
from pymessenger import Bot
from IntentClassifierTrained import *
from NER_tags_tokens import *
from pyknow import *

app = Flask(__name__)

app.config['MONGO_DBNAME'] = 'aipla'
app.config['MONGO_URI'] = 'mongodb://firas:iamthegrandfox991@ds139251.mlab.com:39251/aipla'

PAGE_ACCESS_TOKEN = 'EAAPTPm64SzoBAJZCaMNhEuBaXxdPrj4tOvcbJfFcBdngJq2T9mKH5AKPjopg6gSXrsonu6q7txIxePhIIbiycJVcuIZC59ZAg4DhZCfECAZCf8wTvgeJahqFhk96Y2RMeygZAls4uaxDqyRXGo9KsJNkKOCXKA4IdlgeteXlPFFRROYQU060gh'
bot = Bot(PAGE_ACCESS_TOKEN)
mongo = PyMongo(app)

run_once = 0
messaging_NER= ''

@app.route('/',methods=['GET'])
def verify():
    #webhook verification
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get("hub.verify_token")=="hello":
            return "Verification token mismatch", 403
        return request.args["hub.challenge"],200
    return "Success", 200

@app.route('/',methods=["POST"])
def webhook():
    data = request.get_json()
    if data['object'] == 'page':
        for entry in data['entry']:
            for messaging_event in entry['messaging']:
                #Extract data -- IDs
                sender_id = messaging_event['sender']['id']
                recipient_id = messaging_event['recipient']['id']

                if messaging_event.get('message'):
                    if 'text' in messaging_event['message']:
                        messaging_text = messaging_event['message']['text']
                    else:
                        messaging_text = 'no text'
                    #Echo
                    global run_once
                    
                    if classify(messaging_text) == [('NER',1)]:
                        if "I'm looking for" in messaging_text:
                            messaging_text = re.sub("I'm looking for books about","",messaging_text)
                            run_once = 0
                        elif "I want" in messaging_text:
                            messaging_text = re.sub("I want","",messaging_text)
                            run_once = 0
                        elif "Do you have" in messaging_text:
                            messaging_text = re.sub("Do you have","",messaging_text)
                            run_once = 0
                        
                        messaging_text = messaging_text[1:]
                        print(messaging_text)
                        resNER,TitB,AutB,GenB= NER(messaging_text)
                        bot.send_text_message(sender_id,resNER)

                        ind = messaging_text.find(' by ')
                        ind2 = messaging_text.find(' genre ')

                        if ind == -1:
                            Title = messaging_text
                            Genre = messaging_text
                            Author = messaging_text
                            print(Title)
                        elif ind2 == -1:
                            Title = messaging_text[:ind]
                            Author = messaging_text[ind+4:]
                            print(Title)
                            print(Author)
                        else:
                            Title = messaging_text[:ind]
                            Author = messaging_text[ind+4:ind2]
                            Genre = messaging_text[ind2+7:]
                            print(Title)
                            print(Author)
                            print(Genre)
                        
                        if run_once == 0:
                            recommend = Engine()
                            recommend.reset()
                            recommend.getSenderID(sender_id)
                            if GenB == 1:
                                find('Genre',Genre,sender_id)
                                recommend.getGenre(Genre)
                                recommend.run() 
                            elif TitB == 1 and AutB == 0:
                                find('Title',Title,sender_id)
                                recommend.getTitle(Title)
                                recommend.run() 
                            elif AutB == 1 and TitB == 0:
                                find('Author',Author,sender_id)
                                recommend.getAuthor(Author)
                                recommend.run() 
                            elif TitB == 1 and AutB == 1:
                                find('Title',Title,sender_id)
                                recommend.getTitle(Title)
                                recommend.run() 
                            
                                       
                            run_once = 1

                    else :
                        response_text = response(messaging_text)
                        log(response_text)
                        bot.send_text_message(sender_id,response_text)
                     
    return "ok",200

def find(Kind,Mess,sender_id):
    i = 0
    kind = Kind
    string = Mess
    #print(string)
    book = mongo.db.books
    #print("Found " + str(book.find({kind: {'$regex': string}}).count()) +" books contains the word "+ string)
    bot.send_text_message(sender_id,"Found " + str(book.find({kind: {'$regex': string}}).count()) +" books contains the word " + string)
    #print("showing available books : ")
    bot.send_text_message(sender_id,"showing available books : ")
    for post in book.find({kind: {'$regex': string}}):
        #print(post['Title'] + " by " + post['Author'])
        i += 1
        if i == 5:
            break 
        bot.send_text_message(sender_id,post['Title'] + " by " + post['Author'])
    return 


@app.route('/addall')
def addall():
    book = mongo.db.books
    with open('BOOKDATABASE.csv', 'r') as csvfile:
        books = csv.reader(csvfile, delimiter=',')
        for i in books:
            #print(i[1])
            book.insert({'Title':i[1], 'Author' : i[2], 'Genre': i[3]})
    return 'Added book!'

def log(mess):
    pprint(mess)
    sys.stdout.flush()

def NER(mess):
    string = defStr(mess)
    str2idx=[]
    for word in string.split():
        str2idx.append(words2idxs([word]))

    print(str2idx)
    word2vec = []
    for x in str2idx:
        for y in x:
            word2vec.append(y)
    word2vec = [word2vec]

    a = np.array(word2vec)
    #a = a.reshape(-1,1)
    a.shape
        
    output = sess.run(all_vars,feed_dict={input_batch:np.array(a),lengths:[len(word2vec[0])]})
    Outputlist = list()
    for val in output[0][0]:
        Outputlist.append(val)
            
    Outputlist = [Outputlist]
    print(Outputlist)
    tags , tokens = [],[]
    tokensArr = word2vec

    tags,tokens = predict_tags(tokensArr,Outputlist)
        
    print(tags)
    print(tokens)
    author = ''
    title = ''
    genre = ''

    for i in range(len(tags[0])):
        if tags[0][i] == 'B-title':
            title += tokens[0][i]+" "
            for j in range(len(tags[0])):
                if tags[0][j] == 'I-title':
                    title += tokens[0][j]+" "
    
    for i in range(len(tags[0])):
        if tags[0][i] == 'B-person':
            author += tokens[0][i]+" "
            for j in range(len(tags[0])):
                if tags[0][j] == 'I-person':
                    author += tokens[0][j]+" "
    
    for i in range(len(tags[0])):
        if tags[0][i] == 'B-genre':
            genre += tokens[0][i]+" "
            for j in range(len(tags[0])):
                if tags[0][j] == 'I-genre':
                    genre += tokens[0][j]+" "

    #print(title)
    #print(author)
    #print(genre)

    titleBool=authorBool=genreBool = 0

    if title != '':
        titleBool = 1
    if author != '':
        authorBool = 1
    if genre != '':
        genreBool = 1
    

    response_NER = "Searching for "+title
    if author != '':
        response_NER += "by "+author
    if genre != '':
        response_NER += "genre : "+genre
    
    return response_NER,titleBool,authorBool,genreBool


class Engine(KnowledgeEngine):
    genre = ''
    author  = ''
    title = ''
    book = mongo.db.books
    sender_id = 0

    def getGenre(self,Genre):
        self.genre = Genre 
    def getAuthor(self,Author):
        self.author = Author 
    def getTitle(self,Title):
        self.title = Title
    def getSenderID(self,sender_id):
        self.sender_id = sender_id
        
    @Rule(OR(~Fact(genre=W()),
             ~Fact(author=W()),
             ~Fact(title=W())))
    def defRules(self):
        if self.genre != '':
            self.declare(Fact(Genre=self.genre))
                         
        if self.author != '':
            self.declare(Fact(Author=self.author))
                         
        if self.title != '':
            self.declare(Fact(Title=self.title))
    
    @Rule(AND(Fact(Genre=W()),
              ~Fact(Author=W()),
              ~Fact(Title=W())))
    def showGenre(self):
        bot.send_text_message(self.sender_id,"We recommend you to check this book : ")
        bk = list(mongo.db.books.aggregate([{'$match' : {'Genre': self.genre}}]))
        ran = random.randint(0,len(bk))
        #print(bk[ran]['Title'] + " by " + bk[ran]['Author'])
        #print(self.genre)
        bot.send_text_message(self.sender_id,bk[ran]['Title'] + " by " + bk[ran]['Author'])
    
    @Rule(AND(~Fact(Genre=W()),
              Fact(Author=W()),
              ~Fact(Title=W())))
    def showAuthor(self):
        bot.send_text_message(self.sender_id,"We recommend you to check this book : ")
        bk = list(mongo.db.books.aggregate([{'$match' : {'Author': self.author}}]))
        ran = random.randint(0,len(bk))
        #print(bk[ran]['Title'] + " by " + bk[ran]['Author'])
        #print(self.genre)
        bot.send_text_message(self.sender_id,bk[ran]['Title'] + " by " + bk[ran]['Author'])
    
    @Rule(AND(~Fact(Genre=W()),
              ~Fact(Author=W()),
              Fact(Title=W())))
    def showTitle(self):
        bot.send_text_message(self.sender_id,"We recommend you to check this book : ")
        book = mongo.db.books
        gen = book.find_one({'Title': self.title})
        bk = list(mongo.db.books.aggregate([{'$match' : {'Genre': gen['Genre']}}]))
        ran = random.randint(0,len(bk))
        #print(bk[ran]['Title'] + " by " + bk[ran]['Author'])
        #print(self.genre)
        bot.send_text_message(self.sender_id,bk[ran]['Title'] + " by " + bk[ran]['Author'])
    
    
    



if __name__ == "__main__":
    app.run(debug = True, port = 80)

