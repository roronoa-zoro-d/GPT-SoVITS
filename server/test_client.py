import requests

def send_greeting(name: str):
    url = "http://localhost:8009/greet/"
    response = requests.post(url, json={"name": name})
    if response.status_code == 200:
        return response.json()["message"]
    else:
        return f"Error: {response.status_code}"

if __name__ == "__main__":
    
    greeting = send_greeting('world')
    print(greeting)