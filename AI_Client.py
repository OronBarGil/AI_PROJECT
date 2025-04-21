import socket
import pickle
import argparse
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import io
import threading

class RFClient:
    """
    Client for Random Forest image classifier.
    Sends images to the server for processing.
    """
    
    def __init__(self, host='localhost', port=9000):
        """Initialize the client with host and port"""
        self.host = host
        self.port = port
        self.connected = False
        self.client_socket = None
        
    def connect(self):
        """Connect to the server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            return True
        except Exception as e:
            print(f"Connection error: {str(e)}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Disconnect from the server"""
        if self.client_socket:
            self.client_socket.close()
            self.client_socket = None
        self.connected = False
        
    def send_request(self, request_data):
        """Send a request to the server and receive the response"""
        if not self.connected:
            if not self.connect():
                return {'status': 'error', 'message': 'Not connected to server'}
                
        try:
            # Serialize request data
            data = pickle.dumps(request_data)
            
            # Send message length followed by the message
            message_length = len(data).to_bytes(8, byteorder='big')
            self.client_socket.sendall(message_length)
            self.client_socket.sendall(data)
            
            # Receive response length
            response_length_bytes = self.client_socket.recv(8)
            if not response_length_bytes:
                return {'status': 'error', 'message': 'Connection closed by server'}
                
            response_length = int.from_bytes(response_length_bytes, byteorder='big')
            
            # Receive full response
            chunks = []
            bytes_received = 0
            
            while bytes_received < response_length:
                chunk = self.client_socket.recv(min(4096, response_length - bytes_received))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_received += len(chunk)
                
            response_data = b''.join(chunks)
            
            # Deserialize response
            response = pickle.loads(response_data)
            return response
            
        except Exception as e:
            print(f"Error in send_request: {str(e)}")
            self.disconnect()
            return {'status': 'error', 'message': str(e)}
            
    def send_image(self, image_path):
        """Send an image to the server for classification"""
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
                
            request = {
                'type': 'image',
                'image_data': image_data
            }
                
            return self.send_request(request)
            
        except Exception as e:
            print(f"Error sending image: {str(e)}")
            return {'status': 'error', 'message': str(e)}


class RFClientGUI:
    """
    GUI for the Random Forest Client
    """
    
    def __init__(self, root, host='localhost', port=9000):
        self.root = root
        self.root.title("Random Forest Image Classifier Client")
        self.root.geometry("800x600")
        self.client = RFClient(host, port)
        
        self.image_path = None
        self.image_data = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the GUI widgets"""
        # Top frame for connection status and server info
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(top_frame, text="Server:").pack(side=tk.LEFT, padx=5)
        self.server_label = tk.Label(top_frame, text=f"{self.client.host}:{self.client.port}")
        self.server_label.pack(side=tk.LEFT, padx=5)
        
        self.status_label = tk.Label(top_frame, text="Disconnected", fg="red")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        self.connect_button = tk.Button(top_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(side=tk.RIGHT, padx=5)
        
        # Main content frame
        content_frame = tk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Image display
        left_frame = tk.Frame(content_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image frame
        image_frame = tk.Frame(left_frame, bd=2, relief=tk.SUNKEN)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_label = tk.Label(image_frame, text="No image selected")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Image controls
        image_controls = tk.Frame(left_frame)
        image_controls.pack(fill=tk.X, padx=5, pady=5)
        
        self.select_button = tk.Button(image_controls, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        self.send_image_button = tk.Button(image_controls, text="Classify Image", command=self.send_image, state=tk.DISABLED)
        self.send_image_button.pack(side=tk.RIGHT, padx=5)
        
        # Right side - Results
        right_frame = tk.Frame(content_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results frame
        results_frame = tk.LabelFrame(right_frame, text="Classification Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def toggle_connection(self):
        """Toggle connection to the server"""
        if not self.client.connected:
            self.status_bar.config(text="Connecting to server...")
            
            def connect_thread():
                if self.client.connect():
                    self.root.after(0, self.update_connection_status, True)
                else:
                    self.root.after(0, self.update_connection_status, False)
                    
            threading.Thread(target=connect_thread).start()
        else:
            self.client.disconnect()
            self.update_connection_status(False)
            
    def update_connection_status(self, connected):
        """Update the connection status in the UI"""
        if connected:
            self.status_label.config(text="Connected", fg="green")
            self.connect_button.config(text="Disconnect")
            self.send_image_button.config(state=tk.NORMAL if self.image_path else tk.DISABLED)
            self.status_bar.config(text="Connected to server")
        else:
            self.status_label.config(text="Disconnected", fg="red")
            self.connect_button.config(text="Connect")
            self.send_image_button.config(state=tk.DISABLED)
            self.status_bar.config(text="Disconnected from server")
            
    def select_image(self):
        """Select an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            try:
                self.image_path = file_path
                self.status_bar.config(text=f"Selected image: {os.path.basename(file_path)}")
                
                # Display image in UI
                image = Image.open(file_path)
                
                # Resize image to fit in the display area while preserving aspect ratio
                max_width = 380
                max_height = 300
                width, height = image.size
                
                if width > max_width or height > max_height:
                    # Calculate scale factor to fit within bounds
                    scale = min(max_width / width, max_height / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = image.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage for Tkinter
                photo = ImageTk.PhotoImage(image)
                
                # Update image display
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep a reference to prevent garbage collection
                
                # Enable send button if connected
                if self.client.connected:
                    self.send_image_button.config(state=tk.NORMAL)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_bar.config(text="Failed to load image")
                
    def send_image(self):
        """Send the selected image to the server"""
        if not self.image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
            
        if not self.client.connected:
            messagebox.showwarning("Warning", "Not connected to server")
            return
            
        self.status_bar.config(text="Sending image to server...")
        self.send_image_button.config(state=tk.DISABLED)
        
        def send_thread():
            response = self.client.send_image(self.image_path)
            self.root.after(0, self.handle_response, response)
            
        threading.Thread(target=send_thread).start()
        
    def handle_response(self, response):
        """Handle the response from the server"""
        self.send_image_button.config(state=tk.NORMAL if self.image_path else tk.DISABLED)
        
        if response['status'] == 'success':
            self.status_bar.config(text="Response received from server")
            
            # Update results text
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            if 'prediction' in response:
                pred = response['prediction']
                self.results_text.insert(tk.END, f"Prediction: {pred['predicted_class']}\n")
                self.results_text.insert(tk.END, f"Confidence: {pred['confidence']:.2f}\n\n")
                
                # Display class probabilities if available
                if 'class_probabilities' in pred:
                    self.results_text.insert(tk.END, "Class Probabilities:\n")
                    for cls, prob in pred['class_probabilities'].items():
                        self.results_text.insert(tk.END, f"- {cls}: {prob:.4f}\n")
                
            self.results_text.config(state=tk.DISABLED)
        else:
            self.status_bar.config(text=f"Error: {response.get('message', 'Unknown error')}")
            
            # Display error in results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {response.get('message', 'Unknown error')}")
            self.results_text.config(state=tk.DISABLED)
            
            # Show error dialog
            messagebox.showerror("Error", response.get('message', 'Unknown error'))


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Random Forest Image Classifier Client')
    parser.add_argument('--host', default='localhost', help='Server hostname or IP')
    parser.add_argument('--port', type=int, default=9000, help='Server port')
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Create the main window
    root = tk.Tk()
    app = RFClientGUI(root, host=args.host, port=args.port)
    
    # Run the application
    root.mainloop()


if __name__ == "__main__":
    main()
