from flask import Flask, request, jsonify, abort
import my
import logging
from my import *
import psutil
import time
import sys
import gc
import traceback

start_time = time.time()
app = Flask(__name__)
generator: my.Generator2 = None
log = logging.getLogger('app')

# app.logger.disabled = True
# logging.getLogger('werkzeug').disabled = True

@app.route('/ready')
def ready():
    return 'OK'


@app.route('/generate/<poet_id>', methods=['POST'])
def generate(poet_id):
    try:
        request_data = request.get_json()
        seed = request_data['seed']
        print("Request poet_id=%s, seed=%s" % (poet_id, seed))
        result = generator.generate(poet_id, seed)
        return jsonify({'poem': result.content()})
    except:
        traceback.print_exc(file=sys.stderr)
        traceback.print_exc()
        abort(500)

# HOTFIX
profiler.disabled = True
from gensim.models.callbacks import CallbackAny2Vec
class MyCallback(CallbackAny2Vec):
    pass

if __name__ == '__main__':
    generator = my.Generator2()
    generator.start()
    print("Started in %.3f seconds" % (time.time() - start_time), file=sys.stderr)
    print("Started in %.3f seconds" % (time.time() - start_time))
    gc.set_threshold(100, 1, 2**31-1)
    app.run(host='0.0.0.0', port=8000)