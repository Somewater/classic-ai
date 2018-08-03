from flask import Flask, request, jsonify
from model import Model
import logging

app = Flask(__name__)
model: Model = None
log = logging.getLogger('app')

@app.route('/ready')
def ready():
    return 'OK'


@app.route('/generate/<poet_id>', methods=['POST'])
def generate(poet_id):
    request_data = request.get_json()
    seed = request_data['seed']
    log.info("Request poet_id=%s, seed=%s" % (poet_id, seed))
    generated_poem = model.generate(poet_id, seed)
    return jsonify({'poem': generated_poem})


if __name__ == '__main__':
    model = Model()
    model.start()
    app.run(port=8000)