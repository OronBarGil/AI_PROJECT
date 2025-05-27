import sqlite3
import hashlib
import os
import base64

class DatabaseHandler:
    """
    Handles database operations for user authentication
    """
    
    def __init__(self, db_path='user_database.db'):
        """Initialize the database handler"""
        self.db_path = db_path
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create a table for storing user preferences/settings
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_settings (
            user_id INTEGER PRIMARY KEY,
            last_login TIMESTAMP,
            preferred_server TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def _generate_salt(self):
        """Generate a random salt"""
        return base64.b64encode(os.urandom(32)).decode('utf-8')
    
    def _hash_password(self, password, salt):
        """Hash the password with the given salt using SHA-256"""
        # Combine password and salt
        salted_password = password + salt
        # Hash the salted password
        hash_obj = hashlib.sha256(salted_password.encode())
        return hash_obj.hexdigest()
    
    def register_user(self, username, password, email=None):
        """
        Register a new user
        
        Args:
            username: User's username
            password: User's password (plaintext, will be hashed)
            email: User's email (optional)
            
        Returns:
            tuple: (success, message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if username already exists
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                return False, "Username already exists"
            
            # Generate salt and hash password
            salt = self._generate_salt()
            password_hash = self._hash_password(password, salt)
            
            # Insert new user
            cursor.execute(
                "INSERT INTO users (username, password_hash, salt, email) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, email)
            )
            
            # Get the user ID
            user_id = cursor.lastrowid
            
            # Create user settings entry
            cursor.execute(
                "INSERT INTO user_settings (user_id) VALUES (?)",
                (user_id,)
            )
            
            conn.commit()
            return True, "Registration successful"
            
        except sqlite3.Error as e:
            return False, f"Database error: {e}"
        finally:
            conn.close()
    
    def authenticate_user(self, username, password):
        """
        Authenticate a user
        
        Args:
            username: User's username
            password: User's password (plaintext)
            
        Returns:
            tuple: (success, user_id or error_message)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get user data
            cursor.execute(
                "SELECT id, password_hash, salt FROM users WHERE username = ?",
                (username,)
            )
            user_data = cursor.fetchone()
            
            if not user_data:
                return False, "Invalid username or password"
            
            user_id, stored_hash, salt = user_data
            
            # Verify password
            calculated_hash = self._hash_password(password, salt)
            if calculated_hash != stored_hash:
                return False, "Invalid username or password"
            
            # Update last login time
            cursor.execute(
                "UPDATE user_settings SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,)
            )
            conn.commit()
            
            return True, user_id
            
        except sqlite3.Error as e:
            return False, f"Database error: {e}"
        finally:
            conn.close()
    
    def update_user_server_preference(self, user_id, server_address):
        """Update the user's preferred server"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE user_settings SET preferred_server = ? WHERE user_id = ?",
                (server_address, user_id)
            )
            conn.commit()
            return True
        except sqlite3.Error:
            return False
        finally:
            conn.close()
    
    def get_user_server_preference(self, user_id):
        """Get the user's preferred server"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT preferred_server FROM user_settings WHERE user_id = ?",
                (user_id,)
            )
            result = cursor.fetchone()
            return result[0] if result and result[0] else None
        except sqlite3.Error:
            return None
        finally:
            conn.close()