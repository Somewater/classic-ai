from http.server import BaseHTTPRequestHandler, HTTPServer
import my
import logging
from my import *
import time
import sys
import gc
import traceback
import json

start_time = time.time()
generator: my.Generator2 = None
log = logging.getLogger('app')

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/ready':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
        else:
            self.send_response(404)

    def do_POST(self):
        try:
            poet_id = self.path.split('/generate/')[-1]
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            request_data = json.loads(post_data)
            seed = request_data['seed']
            print(time.asctime(), "Request poet_id=%s, seed=%s" % (poet_id, seed))
            result = generator.generate(poet_id, seed)
            response_data = json.dumps({'poem': result.content()})
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(response_data.encode('utf-8'))
        except:
            traceback.print_exc(file=sys.stderr)
            traceback.print_exc()
            self.send_response(500)
            self.end_headers()

    def log_message(self, format, *args):
        return


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

    logging.basicConfig(level=logging.INFO)
    server_class = HTTPServer
    httpd = HTTPServer(('0.0.0.0', 8000), MyHandler)
    print(time.asctime(), 'Server Starts')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()