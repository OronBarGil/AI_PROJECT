import socket

IP = "127.0.0.1"
PORT = 12345



class Network():
    def __init__(self):
        self.sock = None
    
    def connect(self, address, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((address, port))
        print(f"Connection succeded to server: {address}")



    def disconnect(self):
        self.sock.close()





def main():
    cl1 = Network()
    cl1.connect(IP, PORT)


if __name__ == "__main__":
    main()