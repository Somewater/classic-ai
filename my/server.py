from flask import Flask, request, jsonify, abort
import locale
import my
import logging
from my import *

app = Flask(__name__)
generator: my.Generator1 = None
log = logging.getLogger('app')

@app.route('/ready')
def ready():
    return 'OK'


@app.route('/generate/<poet_id>', methods=['POST'])
def generate(poet_id):
    request_data = request.get_json()
    seed = request_data['seed']
    log.info("Request poet_id=%s, seed=%s" % (poet_id, seed))
    try:
        result = generator.generate(poet_id, seed)
        return jsonify({'poem': result.content()})
    except KeyError:
        abort(404)

# HOTFIX
from gensim.models.callbacks import CallbackAny2Vec
class MyCallback(CallbackAny2Vec):
    pass

if __name__ == '__main__':
    reader = DataReader()
    freq = Frequency(reader)
    ortho = OrthoDict(freq)
    generator = my.Generator2(logging.getLogger('generator'), reader, ortho, freq)
    generator.start()
    app.run(host='0.0.0.0', port=8000)