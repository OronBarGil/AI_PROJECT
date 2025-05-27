import socket
import pickle
import argparse
import os
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from PIL import Image, ImageTk
import threading
import numpy as np
import base64
from SecurityUtils import SecurityUtils


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
        self.user_id = None
        self.username = None
        
        # Security setup
        self.server_public_key = None
        self.private_key = None
        self.public_key = None
        self.session_key = None
        self.session_iv = None
        
    def setup_security(self):
        """Set up encryption keys for secure communication"""
        try:
            # Generate RSA key pair for secure communication
            private_key_pem, public_key_pem = SecurityUtils.generate_rsa_key_pair()
            self.private_key = private_key_pem
            self.public_key = public_key_pem
            return True
        except Exception as e:
            print(f"Failed to set up security: {str(e)}")
            return False

    def connect(self):
        """Connect to the server"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((self.host, self.port))
            self.connected = True
            
            # Security handshake
            if not self.setup_security():
                print("Failed to set up security")
                self.disconnect()
                return False
            
            # Receive server's public key
            response = self.receive_message()
            if response and 'type' in response and response['type'] == 'public_key':
                self.server_public_key = response['key'].encode('utf-8')
                print("Received server public key")
                
                # Send our public key to the server
                self.send_message({
                    'type': 'public_key',
                    'key': self.public_key.decode('utf-8')
                })
                
                # Receive session key from server
                key_response = self.receive_message()
                if key_response and 'type' in key_response and key_response['type'] == 'session_key':
                    # Decrypt session key using our private key
                    encrypted_key = key_response['key']
                    encrypted_iv = key_response['iv']
                    
                    try:
                        decrypted_key = SecurityUtils.decrypt_with_private_key(encrypted_key, self.private_key)
                        decrypted_iv = SecurityUtils.decrypt_with_private_key(encrypted_iv, self.private_key)
                        
                        self.session_key = base64.b64decode(decrypted_key)
                        self.session_iv = base64.b64decode(decrypted_iv)
                        print("Session key established")
                        return True
                    except Exception as e:
                        print(f"Failed to decrypt session key: {str(e)}")
                        self.disconnect()
                        return False
                        
                else:
                    print("Failed to receive session key")
                    self.disconnect()
                    return False
            else:
                print("Failed to receive server public key")
                self.disconnect()
                return False
                
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
        self.user_id = None
        self.username = None
        self.server_public_key = None
        self.session_key = None
        self.session_iv = None
        
    def send_message(self, message_data):
        """Send a message to the server (with encryption if keys are available)"""
        if not self.connected:
            # Attempt to reconnect or handle error
            print("Error: Not connected to server.")
            # Optionally, try self.connect() here, but be careful about loops
            return False # Indicate failure

        try:
            # 1. Serialize message data using pickle
            data_bytes = pickle.dumps(message_data)

            # 2. Encrypt the raw bytes if session key is available
            if self.session_key and self.session_iv:
                encrypted_data = SecurityUtils.encrypt_with_aes(
                    data_bytes,
                    self.session_key,
                    self.session_iv
                )
            else:
                # Send unencrypted if session keys not established (e.g., during handshake)
                # Or handle as an error depending on your security requirements
                print("Warning: Sending unencrypted message (session key not set).")
                encrypted_data = data_bytes # Fallback to unencrypted

            # 3. Send message length followed by the (potentially encrypted) message bytes
            message_length = len(encrypted_data).to_bytes(8, byteorder='big')
            self.client_socket.sendall(message_length)
            self.client_socket.sendall(encrypted_data)

            return True # Indicate success
        except (socket.error, pickle.PicklingError, Exception) as e:
            print(f"Error in send_message: {str(e)}")
            self.disconnect() # Disconnect on send error
            return False

    def receive_message(self):
        """Receive a message from the server (with decryption if keys are available)"""
        if not self.connected or not self.client_socket:
            print("Error: Not connected to server.")
            return None # Indicate error or disconnection

        try:
            # 1. Receive response length
            response_length_bytes = self.client_socket.recv(8)
            if not response_length_bytes or len(response_length_bytes) < 8:
                print("Connection closed by server or incomplete header received.")
                self.disconnect()
                return None

            response_length = int.from_bytes(response_length_bytes, byteorder='big')
            if response_length == 0:
                 print("Received zero length message.")
                 return None # Or handle appropriately

            # 2. Receive full response bytes
            chunks = []
            bytes_received = 0
            self.client_socket.settimeout(10.0) # Add a timeout
            while bytes_received < response_length:
                chunk = self.client_socket.recv(min(4096, response_length - bytes_received))
                if not chunk:
                    print("Connection broken while receiving data.")
                    self.disconnect()
                    return None
                chunks.append(chunk)
                bytes_received += len(chunk)
            self.client_socket.settimeout(None) # Reset timeout
            encrypted_data = b''.join(chunks)

            # 3. Decrypt if session key is available
            if self.session_key and self.session_iv:
                try:
                    decrypted_data_bytes = SecurityUtils.decrypt_with_aes(
                        encrypted_data,
                        self.session_key,
                        self.session_iv
                    )
                except Exception as e: # Catch potential decryption/padding errors
                    print(f"Error decrypting message: {str(e)}")
                    self.disconnect()
                    return None
            else:
                 # Handle unencrypted messages if expected (e.g., during handshake)
                 print("Warning: Received unencrypted message (session key not set).")
                 decrypted_data_bytes = encrypted_data # Assume unencrypted

            # 4. Deserialize response using pickle
            return pickle.loads(decrypted_data_bytes)

        except socket.timeout:
            print("Socket timeout during receive.")
            self.disconnect()
            return None
        except (socket.error, pickle.UnpicklingError, EOFError, Exception) as e:
            print(f"Error in receive_message: {str(e)}")
            self.disconnect()
            return None

    def send_request(self, request_data):
        """Send a request to the server and receive the response"""
        if not self.send_message(request_data):
            return {'status': 'error', 'message': 'Failed to send request'}
        return self.receive_message()
            
    def register(self, username, password, email=None):
        """Register a new user account"""
        request = {
            'type': 'register',
            'username': username,
            'password': password
        }
        if email:
            request['email'] = email
            
        return self.send_request(request)

    def login(self, username, password):
        """Log in with username and password"""
        request = {
            'type': 'login',
            'username': username,
            'password': password
        }
        
        response = self.send_request(request)
        
        if response and response.get('status') == 'success':
            self.user_id = response.get('user_id')
            self.username = username
            
        return response

    def send_image(self, image_path):
        """Send an image to the server for classification"""
        try:
            x_test = []
            img = Image.open(image_path).convert('RGB')
            img = img.resize((32, 32), Image.LANCZOS)  # Resize to 32x32
            img_array = np.array(img) / 255.0  # Normalize to [0,1]
            x_test.append(img_array)
            x_test = np.array(x_test, dtype=np.float32)    
            request = {
                'type': 'image',
                'image_data': x_test
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
        
        # Authentication state
        self.is_authenticated = False
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create the GUI widgets"""
        # Create a notebook with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create the authentication frame
        self.auth_frame = ttk.Frame(self.notebook)
        
        # Create the main application frame
        self.app_frame = ttk.Frame(self.notebook)
        
        # Add the frames to the notebook
        self.notebook.add(self.auth_frame, text="Login/Register")
        self.notebook.add(self.app_frame, text="Image Classification")
        
        # Initially disable the app tab
        self.notebook.tab(1, state="disabled")
        
        # Setup the authentication tab
        self.setup_auth_tab()
        
        # Setup the application tab
        self.setup_app_tab()
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def setup_auth_tab(self):
        """Setup the authentication tab with login and register forms"""
        # Create frames for login and register
        login_frame = ttk.LabelFrame(self.auth_frame, text="Login")
        login_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        register_frame = ttk.LabelFrame(self.auth_frame, text="Register")
        register_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Connection status display
        conn_frame = ttk.Frame(self.auth_frame)
        conn_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(conn_frame, text="Server:").pack(side=tk.LEFT, padx=5)
        self.server_label = ttk.Label(conn_frame, text=f"{self.client.host}:{self.client.port}")
        self.server_label.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(conn_frame, text="Disconnected", foreground="red")
        self.status_label.pack(side=tk.RIGHT, padx=5)
        
        self.connect_button = ttk.Button(conn_frame, text="Connect", command=self.toggle_connection)
        self.connect_button.pack(side=tk.RIGHT, padx=5)
        
        # Login form
        ttk.Label(login_frame, text="Username:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.login_username = ttk.Entry(login_frame, width=30)
        self.login_username.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(login_frame, text="Password:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.login_password = ttk.Entry(login_frame, width=30, show="*")
        self.login_password.grid(row=1, column=1, padx=10, pady=10)
        
        login_button = ttk.Button(login_frame, text="Login", command=self.login_user)
        login_button.grid(row=2, column=1, padx=10, pady=20, sticky=tk.E)
        
        # Register form
        ttk.Label(register_frame, text="Username:").grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.reg_username = ttk.Entry(register_frame, width=30)
        self.reg_username.grid(row=0, column=1, padx=10, pady=10)
        
        ttk.Label(register_frame, text="Password:").grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.reg_password = ttk.Entry(register_frame, width=30, show="*")
        self.reg_password.grid(row=1, column=1, padx=10, pady=10)
        
        ttk.Label(register_frame, text="Confirm Password:").grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.reg_confirm_password = ttk.Entry(register_frame, width=30, show="*")
        self.reg_confirm_password.grid(row=2, column=1, padx=10, pady=10)
        
        ttk.Label(register_frame, text="Email (optional):").grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        self.reg_email = ttk.Entry(register_frame, width=30)
        self.reg_email.grid(row=3, column=1, padx=10, pady=10)
        
        register_button = ttk.Button(register_frame, text="Register", command=self.register_user)
        register_button.grid(row=4, column=1, padx=10, pady=20, sticky=tk.E)
        
    def setup_app_tab(self):
        """Setup the main application tab with image classification functionality"""
        # Top frame for user info
        top_frame = ttk.Frame(self.app_frame)
        top_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(top_frame, text="User:").pack(side=tk.LEFT, padx=5)
        self.user_label = ttk.Label(top_frame, text="Not logged in")
        self.user_label.pack(side=tk.LEFT, padx=5)
        
        self.logout_button = ttk.Button(top_frame, text="Logout", command=self.logout_user)
        self.logout_button.pack(side=tk.RIGHT, padx=5)
        
        # Main content frame
        content_frame = ttk.Frame(self.app_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Image display
        left_frame = ttk.Frame(content_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Image frame
        image_frame = ttk.Frame(left_frame, relief=tk.SUNKEN, borderwidth=2)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.image_label = ttk.Label(image_frame, text="No image selected")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Image controls
        image_controls = ttk.Frame(left_frame)
        image_controls.pack(fill=tk.X, padx=5, pady=5)
        
        self.select_button = ttk.Button(image_controls, text="Select Image", command=self.select_image)
        self.select_button.pack(side=tk.LEFT, padx=5)
        
        self.send_image_button = ttk.Button(image_controls, text="Classify Image", 
                                         command=self.send_image, state=tk.DISABLED)
        self.send_image_button.pack(side=tk.RIGHT, padx=5)
        
        # Right side - Results
        right_frame = ttk.Frame(content_frame, width=400)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results frame
        results_frame = ttk.LabelFrame(right_frame, text="Classification Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, wrap=tk.WORD)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.results_text.config(state=tk.DISABLED)
    
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
            
            # Reset authentication state
            self.is_authenticated = False
            self.notebook.tab(1, state="disabled")
            self.user_label.config(text="Not logged in")
            
    def update_connection_status(self, connected):
        """Update the connection status in the UI"""
        if connected:
            self.status_label.config(text="Connected", foreground="green")
            self.connect_button.config(text="Disconnect")
            self.status_bar.config(text="Connected to server")
        else:
            self.status_label.config(text="Disconnected", foreground="red")
            self.connect_button.config(text="Connect")
            self.send_image_button.config(state=tk.DISABLED)
            self.status_bar.config(text="Disconnected from server")
    
    def login_user(self):
        """Login the user"""
        if not self.client.connected:
            messagebox.showwarning("Not Connected", "Please connect to the server first")
            return
            
        username = self.login_username.get().strip()
        password = self.login_password.get().strip()
        
        if not username or not password:
            messagebox.showwarning("Missing Information", "Please enter both username and password")
            return
            
        self.status_bar.config(text="Logging in...")
        
        def login_thread():
            response = self.client.login(username, password)
            self.root.after(0, self.handle_login_response, response)
            
        threading.Thread(target=login_thread).start()
    
    def handle_login_response(self, response):
        """Handle the login response from the server"""
        if response and response.get('status') == 'success':
            self.is_authenticated = True
            self.user_label.config(text=f"{self.client.username}")
            self.status_bar.config(text="Logged in successfully")
            
            # Enable the app tab and switch to it
            self.notebook.tab(1, state="normal")
            self.notebook.select(1)
            
            # Clear login form
            self.login_username.delete(0, tk.END)
            self.login_password.delete(0, tk.END)
            
            # Enable image classification if an image is already selected
            if self.image_path:
                self.send_image_button.config(state=tk.NORMAL)
        else:
            error_msg = response.get('message', 'Login failed') if response else "No response from server"
            self.status_bar.config(text=f"Login failed: {error_msg}")
            messagebox.showerror("Login Failed", error_msg)
    
    def register_user(self):
        """Register a new user"""
        if not self.client.connected:
            messagebox.showwarning("Not Connected", "Please connect to the server first")
            return
            
        username = self.reg_username.get().strip()
        password = self.reg_password.get().strip()
        confirm_password = self.reg_confirm_password.get().strip()
        email = self.reg_email.get().strip()
        
        if not username or not password:
            messagebox.showwarning("Missing Information", "Please enter username and password")
            return
            
        if password != confirm_password:
            messagebox.showwarning("Password Mismatch", "Passwords do not match")
            return
            
        self.status_bar.config(text="Registering new user...")
        
        def register_thread():
            response = self.client.register(username, password, email if email else None)
            self.root.after(0, self.handle_register_response, response)
            
        threading.Thread(target=register_thread).start()
    
    def handle_register_response(self, response):
        """Handle the registration response from the server"""
        if response and response.get('status') == 'success':
            self.status_bar.config(text="Registration successful")
            messagebox.showinfo("Registration Successful", "Your account has been created successfully. You can now login.")
            
            # Clear registration form
            self.reg_username.delete(0, tk.END)
            self.reg_password.delete(0, tk.END)
            self.reg_confirm_password.delete(0, tk.END)
            self.reg_email.delete(0, tk.END)
        else:
            error_msg = response.get('message', 'Registration failed') if response else "No response from server"
            self.status_bar.config(text=f"Registration failed: {error_msg}")
            messagebox.showerror("Registration Failed", error_msg)
    
    def logout_user(self):
        """Logout the current user"""
        # Reset authentication state
        self.client.user_id = None
        self.client.username = None
        self.is_authenticated = False
        
        # Update UI
        self.user_label.config(text="Not logged in")
        self.notebook.tab(1, state="disabled")
        self.notebook.select(0)
        
        self.status_bar.config(text="Logged out successfully")
            
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
                
                # Enable send button if connected and authenticated
                if self.client.connected and self.is_authenticated:
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
            
        if not self.is_authenticated:
            messagebox.showwarning("Warning", "Please login first")
            return
            
        self.status_bar.config(text="Sending image to server...")
        self.send_image_button.config(state=tk.DISABLED)
        
        def send_thread():
            response = self.client.send_image(self.image_path)
            self.root.after(0, self.handle_response, response)
            
        threading.Thread(target=send_thread).start()
        
    def handle_response(self, response):
        """Handle the response from the server"""
        self.send_image_button.config(state=tk.NORMAL if self.image_path and self.is_authenticated else tk.DISABLED)
        
        if response and response.get('status') == 'success':
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
                    for cls, prob in sorted(pred['class_probabilities'].items(), key=lambda x: x[1], reverse=True):
                        self.results_text.insert(tk.END, f"- {cls}: {prob:.4f}\n")
                
            self.results_text.config(state=tk.DISABLED)
            
        else:
            error_msg = response.get('message', 'Unknown error') if response else "No response from server"
            self.status_bar.config(text=f"Error: {error_msg}")
            
            # Display error in results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error: {error_msg}")
            self.results_text.config(state=tk.DISABLED)
            
            # Show error dialog
            messagebox.showerror("Error", error_msg)


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