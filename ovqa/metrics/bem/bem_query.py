"""

"""
import argparse
import numpy as np
import requests
import time

from packg import format_exception


def query_bem(data: list[dict[str, str]], server="localhost:5000") -> list[float]:
    """

    Args:
        data: query with list of dict like
            [
                {
                    "question": "why is the sky blue",
                    "reference": "light scattering",
                    "candidate": "scattering of light",
                },
                ...
            ]
        server: server and port where the bem_server.py flask app is running

    Returns:
        list of scores in 0-1 range, same length as input data

    """
    url = f"http://{server}/query"
    tries_left = 3
    while True:
        try:
            response = requests.post(url, json=data)
            decoded_data = response.json()
            return decoded_data
        except Exception as e:
            print(f"Bem querying failed: {format_exception(e)}")
            if tries_left == 0:
                print(f"Giving up!")
                raise ConnectionError(f"Make sure BEM server is running at {url}") from e
            print(f"Will try {tries_left} more times...")
            tries_left -= 1
            time.sleep(1)
            continue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="localhost:5000")
    args = parser.parse_args()

    inputs = [
        {
            "question": "why is the sky blue",
            "reference": "light scattering",
            "candidate": "scattering of light",
        },
        {
            "question": "why is the banana askew",
            "reference": "it is growing towards the light",
            "candidate": "because this protects them against elephants",
        },
        {
            "question": "are there any elephants in the room",
            "reference": "yes",
            "candidate": "no",
        },
    ]
    print(f"Querying {args.server} with inputs:\n{inputs}")
    response = query_bem(inputs, server=args.server)
    print(f"Response: {response}")
    np.testing.assert_allclose(response, [0.9892, 0.0709, 0.8647], atol=1e-2)
    print("Success!")


if __name__ == "__main__":
    main()
