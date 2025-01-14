import socket

IP = "0.0.0.0"
PORT = 12345



class Network():
    def __init__(self):
        self.sock = None
    
    def connect(self, address, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((address, port))
        self.sock.listen(5)


    def disconnect(self):
        self.sock.close()






def main():
    server = Network()
    server.connect(IP, PORT)
    while True:
        print("Server starting")
        client_socket, client_address = server.sock.accept()
        print(f"Connection succeded to client: {client_address}")


if __name__ == "__main__":
    main()