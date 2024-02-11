"""
See README.md "Installing and running BEM metric"
"""
import argparse
from flask import Flask, request, jsonify

try:
    from .bem_tensorflow import BEM
except ImportError:
    from bem_tensorflow import BEM

app = Flask(__name__)
model = BEM()


@app.route("/query", methods=["POST"])
def query_model():
    data = request.json
    print(f"Got data: {data}")
    response = model(data)  # Assume your model has a query method
    return jsonify(response.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port)  # Allow connections from outside


if __name__ == "__main__":
    main()
