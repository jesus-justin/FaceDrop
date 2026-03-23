from flask import Flask, request, jsonify
import requests

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/claude', methods=['POST'])
def proxy():
    body = request.json
    key  = body.pop('apiKey', '')
    try:
        r = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers={
                'Content-Type':      'application/json',
                'x-api-key':         key,
                'anthropic-version': '2023-06-01'
            },
            json=body,
            timeout=30
        )
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5500, debug=True)
