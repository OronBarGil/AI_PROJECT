import base64
import os
from cryptography.hazmat.primitives.asymmetric import rsa, padding as rsa_padding
from cryptography.hazmat.primitives import hashes, serialization, padding as sym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class SecurityUtils:
    """
    Utility class for handling encryption/decryption for client-server communication
    (Updated AES functions)
    """

    @staticmethod
    def generate_rsa_key_pair():
        """
        Generate an RSA key pair

        Returns:
            tuple: (private_key, public_key) both in PEM format
        """
        # Generate a private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # Get private key in PEM format
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        # Get public key in PEM format
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        return private_pem, public_pem

    @staticmethod
    def encrypt_with_public_key(message, public_key_pem):
        """
        Encrypt a message using an RSA public key

        Args:
            message: String message to encrypt
            public_key_pem: Public key in PEM format (bytes)

        Returns:
            str: Base64 encoded encrypted message
        """
        # Load the public key
        public_key = serialization.load_pem_public_key(
            public_key_pem,
            backend=default_backend()
        )

        # Encrypt the message
        encrypted = public_key.encrypt(
            message.encode('utf-8'), # Ensure message is bytes
            rsa_padding.OAEP(
                mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return base64.b64encode(encrypted).decode('utf-8') # Return base64 string

    @staticmethod
    def decrypt_with_private_key(encrypted_message_b64, private_key_pem):
        """
        Decrypt a message using an RSA private key

        Args:
            encrypted_message_b64: Base64 encoded encrypted message (string)
            private_key_pem: Private key in PEM format (bytes)

        Returns:
            str: Decrypted message
        """
        # Load the private key
        private_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
            backend=default_backend()
        )

        # Decode from base64 first
        encrypted_message_bytes = base64.b64decode(encrypted_message_b64)

        # Decrypt the message
        decrypted = private_key.decrypt(
            encrypted_message_bytes,
            rsa_padding.OAEP(
                mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )

        return decrypted.decode('utf-8') # Return decoded string

    @staticmethod
    def generate_aes_key():
        """
        Generate a random AES key and initialization vector

        Returns:
            tuple: (key, iv) both as bytes
        """
        key = os.urandom(32)  # 256-bit key
        iv = os.urandom(16)   # 128-bit IV for CBC
        return key, iv

    @staticmethod
    def encrypt_with_aes(data_bytes, key, iv):
        """
        Encrypt bytes using AES-256-CBC with PKCS7 padding.

        Args:
            data_bytes: Plaintext data (bytes) to encrypt.
            key: AES key (bytes).
            iv: Initialization vector (bytes).

        Returns:
            bytes: Encrypted ciphertext.
        """
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        # Use PKCS7 padding from the library
        padder = sym_padding.PKCS7(algorithms.AES.block_size).padder()
        padded_data = padder.update(data_bytes) + padder.finalize()
        encrypted = encryptor.update(padded_data) + encryptor.finalize()
        return encrypted

    @staticmethod
    def decrypt_with_aes(encrypted_message_bytes, key, iv):
        """
        Decrypt bytes using AES-256-CBC with PKCS7 padding.

        Args:
            encrypted_message_bytes: Encrypted ciphertext (bytes).
            key: AES key (bytes).
            iv: Initialization vector (bytes).

        Returns:
            bytes: Decrypted plaintext data.
        """
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        # Use PKCS7 unpadding from the library
        unpadder = sym_padding.PKCS7(algorithms.AES.block_size).unpadder()
        decrypted_padded = decryptor.update(encrypted_message_bytes) + decryptor.finalize()
        decrypted = unpadder.update(decrypted_padded) + unpadder.finalize()
        return decrypted