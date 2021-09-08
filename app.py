from flask import Flask,render_template, request
import torch
import  random
import  json
from model import  NeuralNet
from nltk_utils import tokenize,stem,bag_of_words
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/get")
def  get_bot_response():
    with open('intents.json', 'r') as json_data:
        intents = json.load(json_data)
    userText = request.args.get('msg')
    data = torch.load("data.pth")

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    all_words = data['all_words']
    tags = data['tags']
    model_state = data["model_state"]
    bot_name = "Sam"
    model = NeuralNet(input_size, hidden_size, output_size)
    model.load_state_dict(model_state)
    model.eval()


    sentence = tokenize(userText)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                res = random.choice(intent['responses'])
    else:
        res="I do not understand..."

    return str(res)


if __name__ == "__main__":
    app.run()
