from flask import Flask, request, jsonify
import locale
import my

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
    generated_poem = generator.generate(poet_id, seed)
    return jsonify({'poem': generated_poem})


if __name__ == '__main__':
    generator = my.Generator1(logging.getLogger('generator'))
    generator.start()
    app.run(port=8000)