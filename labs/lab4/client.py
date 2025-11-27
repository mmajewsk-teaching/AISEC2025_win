import requests

API_URL = "http://localhost:8001/chat"

if __name__ == "__main__":
    print("Streaming Chatbot (type '\\q' to quit)")
    while True:
        user_input = input("\n>>: ")
        if user_input == "\\q":
            print("Bye!")
            break

        print("\nAssistant: ", end="", flush=True)
        response = requests.post(API_URL, json={"message": user_input}, stream=True)
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                print(chunk, end="", flush=True)
        print()
