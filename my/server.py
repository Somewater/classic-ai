from flask import Flask, request, jsonify, abort
import my
import logging
from my import *
import psutil
import time

start_time = time.time()
app = Flask(__name__)
generator: my.Generator2 = None
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
profiler.disabled = True
from gensim.models.callbacks import CallbackAny2Vec
class MyCallback(CallbackAny2Vec):
    pass

if __name__ == '__main__':
    generator = my.Generator2()
    generator.start()
    generator.log.info("Started in %.3f seconds" % (time.time() - start_time))
    generator.log.info("MEM: %s" % repr(psutil.virtual_memory()))
    generator.log.info("SWAP: %s" % repr(psutil.swap_memory()))
    generator.log.info("CPU(%d): %s" % (psutil.cpu_count(), repr(psutil.cpu_freq())))
    app.run(host='0.0.0.0', port=8000)