from flask import Flask, request

app = Flask(__name__)

@app.route('/api-endpoint', methods=['POST','GET'])
def handle_post_request():
    # Get the JSON or form data from the request
    data = request.get_json() if request.is_json else request.form
    print("Received POST request with data:", data)

    #here the script to analyze the flows and send them to the siem

    # You can return a response if needed
    return "Request received!", 200

if __name__ == '__main__':
    # Run the app in debug mode for development
    app.run(host='0.0.0.0', port=80, debug=True)