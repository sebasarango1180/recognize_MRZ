from flask import Flask
import matcher
import os

app = Flask(__name__)

@app.route('/<path>', methods=['GET'])  # To include static files.
def distribute_img(path):
    
    m = matcher.Matcher()
    if path.split('.')[0][-4::] == 'back':
        return m.text_extraction_MRZ(path), 200

    elif path.split('.')[0][-5::] == 'front':
        return '', 204

    else:
        '', 205

if __name__ == "__main__":

    port = int(os.environ.get('PORT', 5000))
    #socketio.run(app, host='0.0.0.0', port=port)
    app.run(host='0.0.0.0', port=port, debug=True)