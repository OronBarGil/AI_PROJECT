import socket
import numpy as np
import pickle
import threading
import os
import json
import time
import base64
from PIL import Image
import io
from Random_Forest import RandomForestClassifier
import cv2
from io import BytesIO

class RFServer:
    """
    Server for Random Forest image classifier.
    Hosts the AI model and processes client requests.
    """
    
    def __init__(self, host='localhost', port=9000, model_path='animal_rf_model.npy', 
             train_dir='train', test_dir='test'):
        """Initialize the server with host, port, model path and data directories"""
        self.host = host
        self.port = port
        self.model_path = model_path
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.server_socket = None
        self.model = None
        self.selected_indices = None
        self.class_names = ['donkey', 'horse', 'zebra'] 
        self.running = False
        
    def load_model(self):
        """Load the Random Forest model from disk"""
        print("Loading model...")
        if os.path.exists(self.model_path):
            model_data = np.load(self.model_path, allow_pickle=True).item()
            self.model = RandomForestClassifier()
            self.model.base_learner_list = model_data['base_learner_list']
            self.model.feature_importances = model_data['feature_importances']
            self.selected_indices = model_data['selected_indices']
            print(f"Model loaded successfully with accuracy: {model_data.get('test_accuracy', 'N/A')}")
        else:
            print(f"Model not found at {self.model_path}. Starting model training...")
            self.train_model()
    

    def augment_training_data(self, features, labels, augmentation_factor=0.5):
        """
        Augment training data by creating synthetic examples through feature perturbation
        
        Args:
            features: Feature array
            labels: Label array
            augmentation_factor: Float between 0-1 determining what fraction of original data size to add
            
        Returns:
            augmented_features, augmented_labels
        """
        n_samples, n_features = features.shape
        n_augmented = int(n_samples * augmentation_factor)
        
        # Initialize augmented arrays
        augmented_features = np.zeros((n_samples + n_augmented, n_features), dtype=features.dtype)
        augmented_labels = np.zeros(n_samples + n_augmented, dtype=labels.dtype)
        
        # Copy original data
        augmented_features[:n_samples] = features
        augmented_labels[:n_samples] = labels
        
        # Create synthetic examples based on existing ones
        for i in range(n_augmented):
            # Select a random example
            idx = np.random.randint(0, n_samples)
            original_feature = features[idx]
            original_label = labels[idx]
            
            # Create a perturbed version
            perturbed_feature = original_feature.copy()

            # Random perturbation strategies:
            strategy = np.random.choice(['jitter', 'swap', 'interpolate'])
            
            if strategy == 'jitter':
                # Add random noise to features
                noise_factor = 0.05
                noise = np.random.normal(0, noise_factor, n_features)
                perturbed_feature += noise
                
            elif strategy == 'swap':
                # Swap random feature segments
                if n_features >= 8:  # Only if we have enough features
                    seg_size = n_features // 8
                    seg1_start = np.random.randint(0, n_features - seg_size)
                    seg2_start = np.random.randint(0, n_features - seg_size)
                    
                    # Avoid overlapping segments
                    while abs(seg1_start - seg2_start) < seg_size:
                        seg2_start = np.random.randint(0, n_features - seg_size)
                        
                    # Swap segments
                    temp = perturbed_feature[seg1_start:seg1_start+seg_size].copy()
                    perturbed_feature[seg1_start:seg1_start+seg_size] = perturbed_feature[seg2_start:seg2_start+seg_size]
                    perturbed_feature[seg2_start:seg2_start+seg_size] = temp
                    
            elif strategy == 'interpolate':
                # Interpolate between this example and another from same class
                same_class_indices = np.where(labels == original_label)[0]
                if len(same_class_indices) > 1:
                    other_idx = np.random.choice(same_class_indices)
                    while other_idx == idx:  # Ensure we pick a different example
                        other_idx = np.random.choice(same_class_indices)
                        
                    # Linear interpolation
                    alpha = np.random.uniform(0.2, 0.8)
                    perturbed_feature = alpha * original_feature + (1-alpha) * features[other_idx]
            
            # Store augmented example
            augmented_features[n_samples + i] = perturbed_feature
            augmented_labels[n_samples + i] = original_label
        
        return augmented_features, augmented_labels

    def train_model(self):
        """Train a Random Forest model with optimized parameters"""
        try:
                # Load and preprocess the data
            print("Loading and preprocessing training data...")
            X_train, y_train, X_test, y_test = self.load_custom_dataset(self.train_dir, self.test_dir)
            
            # Extract features
            print("Extracting features from training images...")
            train_features = self.extract_enhanced_features(X_train)
            test_features = self.extract_enhanced_features(X_test)
            
            # Feature selection - use more features now
            print("Performing feature selection...")
            self.selected_indices = self.select_features(train_features, y_train, n_features=80)
            
            # Apply feature selection
            train_features_selected = train_features[:, self.selected_indices]
            test_features_selected = test_features[:, self.selected_indices]
            
            # Data augmentation for training
            print("Performing data augmentation...")
            augmented_features, augmented_labels = self.augment_training_data(
                train_features_selected, y_train
            )
            
            # Train the model with optimized parameters
            print("Training Random Forest model with optimized parameters...")
            self.model = RandomForestClassifier(
                n_base_learner=100,       # More trees for better ensemble performance
                max_depth=15,             # Allow deeper trees for complex patterns
                min_samples_leaf=2,       # Prevent overfitting while maintaining detail
                min_information_gain=0.001, # Lower threshold to capture subtle patterns
                numb_of_features_splitting='sqrt',  # Standard for random forests
                bootstrap_sample_size=int(0.8 * augmented_features.shape[0])  # Use 80% of data per tree
            )
            
            # Train on augmented data
            self.model.train(augmented_features, augmented_labels)
            
            # Evaluate the model
            print("Evaluating model performance...")
            predictions = self.model.predict(test_features_selected)
            accuracy = np.mean(predictions == y_test)
            print(f"Model accuracy on test set: {accuracy:.4f}")
            
            # Class-specific performance
            class_accuracies = {}
            for class_idx in range(10):
                class_mask = (y_test == class_idx)
                if np.sum(class_mask) > 0:
                    class_acc = np.mean(predictions[class_mask] == y_test[class_mask])
                    class_accuracies[self.class_names[class_idx]] = class_acc
                    print(f"Accuracy for class '{self.class_names[class_idx]}': {class_acc:.4f}")
            
            # Save the model
            print(f"Saving model to {self.model_path}...")
            model_data = {
                'base_learner_list': self.model.base_learner_list,
                'feature_importances': self.model.feature_importances,
                'selected_indices': self.selected_indices,
                'test_accuracy': accuracy,
                'class_accuracies': class_accuracies
            }
            
            np.save(self.model_path, model_data)
            print("Model training completed successfully!")
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            raise RuntimeError(f"Failed to train model: {str(e)}")
    
    def download_dataset(self, dataset_dir):
        """Download CIFAR-10 dataset"""
        import urllib.request
        import tarfile
        
        print("Downloading CIFAR-10 dataset...")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # URL for CIFAR-10 dataset
        url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
        tar_file = os.path.join(dataset_dir, "cifar-10-python.tar.gz")
        
        # Download file
        urllib.request.urlretrieve(url, tar_file)
        
        # Extract files
        with tarfile.open(tar_file, "r:gz") as tar:
            tar.extractall(path=dataset_dir)
            
        print("Download and extraction completed.")
    
    def load_custom_dataset(self, train_dir, test_dir):
        """Load custom dataset from directories containing class folders"""
        class_names = ['donkey', 'horse', 'zebra']
        self.class_names = class_names
        
        # Initialize data containers
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        # Load training data
        print("Loading training data...")
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        # Load and preprocess image
                        image = Image.open(img_path).convert('RGB')
                        image = image.resize((32, 32), Image.LANCZOS)
                        img_array = np.array(image) / 255.0  # Normalize to [0,1]
                        
                        X_train.append(img_array)
                        y_train.append(class_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
        
        # Load test data
        print("Loading test data...")
        for class_idx, class_name in enumerate(class_names):
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
                
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    try:
                        # Load and preprocess image
                        image = Image.open(img_path).convert('RGB')
                        image = image.resize((32, 32), Image.LANCZOS)
                        img_array = np.array(image) / 255.0  # Normalize to [0,1]
                        
                        X_test.append(img_array)
                        y_test.append(class_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {str(e)}")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"Classes: {', '.join(class_names)}")
        
        return X_train, y_train, X_test, y_test
    
    def select_features(self, features, labels, n_features=80):
        """Select the most informative features using multiple criteria"""
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        
        print(f"Starting feature selection from {features.shape[1]} features...")
        
        # Get feature importance using ANOVA F-values
        selector_f = SelectKBest(f_classif, k=n_features)
        selector_f.fit(features, labels)
        f_scores = selector_f.scores_
        
        # Also use mutual information for non-linear relationships
        selector_mi = SelectKBest(mutual_info_classif, k=n_features)
        selector_mi.fit(features, labels)
        mi_scores = selector_mi.scores_
        
        # Normalize scores
        f_scores = (f_scores - np.min(f_scores)) / (np.max(f_scores) - np.min(f_scores) + 1e-10)
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-10)
        
        # Combine scores (giving equal weight to both methods)
        combined_scores = 0.5 * f_scores + 0.5 * mi_scores
        
        # Select top features based on combined score
        selected_indices = np.argsort(combined_scores)[-n_features:]
        
        # Also check for highly correlated features and possibly remove some
        selected_features = features[:, selected_indices]
        correlation_matrix = np.corrcoef(selected_features.T)
        
        # Identify highly correlated pairs
        high_corr_threshold = 0.95
        highly_correlated_pairs = []
        
        for i in range(len(selected_indices)):
            for j in range(i+1, len(selected_indices)):
                if np.abs(correlation_matrix[i, j]) > high_corr_threshold:
                    # Keep track of correlated pairs and their importance scores
                    highly_correlated_pairs.append((i, j, combined_scores[selected_indices[i]], 
                                                combined_scores[selected_indices[j]]))
        
        # Remove the less important feature in highly correlated pairs
        indices_to_remove = set()
        for i, j, score_i, score_j in highly_correlated_pairs:
            if score_i < score_j:
                indices_to_remove.add(i)
            else:
                indices_to_remove.add(j)
        
        # Create final indices list excluding highly correlated features
        final_indices = [idx for idx_pos, idx in enumerate(selected_indices) 
                        if idx_pos not in indices_to_remove]
        
        # If we removed too many, add back some features based on individual importance
        remaining_features = n_features - len(final_indices)
        if remaining_features > 0:
            # Get all indices sorted by importance, excluding already selected ones
            all_indices_by_importance = np.argsort(combined_scores)[::-1]
            additional_indices = [idx for idx in all_indices_by_importance 
                                if idx not in final_indices][:remaining_features]
            final_indices.extend(additional_indices)
        
        print(f"Selected {len(final_indices)} features")
        return np.array(final_indices)
    
    def extract_enhanced_features(self, images):
        """Extract improved features from images for better classification"""
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        
        n_samples = images.shape[0]
        # Increase feature vector size for more discriminative power
        features = np.zeros((n_samples, 128), dtype=np.float32)
        
        for i in range(n_samples):
            img = images[i]
            
            # Ensure img is in the correct format for OpenCV (uint8)
            img_uint8 = (img * 255).astype(np.uint8)
            
            # Convert to different color spaces for more information
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
            lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB).astype(np.float32) / 255.0
            
            # 1. Enhanced Color features (24)
            # RGB statistics (more granular)
            features[i, 0:3] = np.mean(img, axis=(0, 1))  # RGB means
            features[i, 3:6] = np.std(img, axis=(0, 1))   # RGB stds
            features[i, 6:9] = np.percentile(img, 75, axis=(0, 1)) - np.percentile(img, 25, axis=(0, 1))  # RGB IQR
            
            # HSV and LAB statistics (better for color perception)
            features[i, 9:12] = np.mean(hsv, axis=(0, 1))  # HSV means
            features[i, 12:15] = np.std(hsv, axis=(0, 1))  # HSV stds
            features[i, 15:18] = np.mean(lab, axis=(0, 1))  # LAB means
            features[i, 18:21] = np.std(lab, axis=(0, 1))  # LAB stds
            
            # Color histograms with better binning
            for c in range(3):
                hist = np.histogram(img[:,:,c], bins=8, range=(0, 1))[0]
                features[i, 21+c] = np.argmax(hist)  # Dominant color bin
            
            # 2. Improved Texture features (20)
            # Multi-scale Gabor filter responses (simplified) for texture
            ksize = 5
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel = cv2.getGaborKernel((ksize, ksize), 2.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features[i, 24 + int(theta/(np.pi/4))*2] = np.mean(filtered)
                features[i, 25 + int(theta/(np.pi/4))*2] = np.std(filtered)
            
            # LBP-inspired features (improved)
            lbp_features = np.zeros(8)
            center = gray[1:-1, 1:-1]
            neighbors = [
                gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:],
                gray[1:-1, :-2],                  gray[1:-1, 2:],
                gray[2:, :-2], gray[2:, 1:-1], gray[2:, 2:]
            ]
            for j, n in enumerate(neighbors):
                lbp_features[j] = np.mean(n > center)
            
            features[i, 32:40] = lbp_features
            
            # GLCM-inspired texture (improved)
            # Simplified co-occurrence metrics
            for offset_x, offset_y in [(1,0), (0,1), (1,1), (1,-1)]:
                if offset_y < 0:
                    rolled = np.roll(gray, -offset_y, axis=0)
                    rolled[0:abs(offset_y), :] = 0
                else:
                    rolled = np.roll(gray, offset_y, axis=0)
                    if offset_y > 0:
                        rolled[-offset_y:, :] = 0
                        
                if offset_x > 0:
                    rolled = np.roll(rolled, offset_x, axis=1)
                    rolled[:, -offset_x:] = 0
                
                # Compute contrast measure for each offset
                idx = 40 + (offset_x + 1) * (offset_y + 1) % 4
                features[i, idx] = np.mean(np.abs(gray - rolled))
                features[i, idx+1] = np.std(np.abs(gray - rolled))
            
            # 3. Enhanced edge features (16)
            # Multi-scale edge detection
            gray_uint8 = (gray * 255).astype(np.uint8)
            for j, (low, high) in enumerate([(25, 75), (50, 150), (100, 200), (150, 250)]):
                edges = cv2.Canny(gray_uint8, low, high)
                features[i, 48+j*2] = edges.mean() / 255.0
                # Edge direction histogram
                if j == 1:  # Only for mid-level edges to save computation
                    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
                    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
                    mag, ang = cv2.cartToPolar(gx, gy)
                    edges_mask = (edges > 0).astype(np.float32)
                    weighted_angles = ang * edges_mask
                    
                    # 8-bin angle histogram for better direction sensitivity
                    ang_bins = np.histogram(weighted_angles[edges_mask > 0], bins=8, range=(0, np.pi))[0]
                    if np.sum(ang_bins) > 0:
                        features[i, 56:64] = ang_bins / np.sum(ang_bins)
                    
            # 4. Improved spatial features (24)
            # Grid-based features (3x3 grid)
            h, w = gray.shape
            h_step, w_step = h // 3, w // 3
            
            idx = 64
            for i_h in range(3):
                for i_w in range(3):
                    region = gray[i_h*h_step:(i_h+1)*h_step, i_w*w_step:(i_w+1)*w_step]
                    features[i, idx] = np.mean(region)
                    features[i, idx+1] = np.std(region)
                    idx += 2
            
            # 5. Scale-space features (8)
            # Multi-scale analysis using Gaussian pyramids
            curr_img = gray.copy()
            for j in range(4):
                features[i, 88+j*2] = np.mean(curr_img)
                features[i, 89+j*2] = np.std(curr_img)
                curr_img = cv2.pyrDown(curr_img)
                
            # 6. HOG-inspired features (12)
            # Simplified HOG using gradient histograms from regions
            regions_h, regions_w = 2, 2
            cell_h, cell_w = h // regions_h, w // regions_w
            
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)
            
            idx = 96
            bins = 3  # 3 orientation bins per region
            
            for i_h in range(regions_h):
                for i_w in range(regions_w):
                    region_mag = mag[i_h*cell_h:(i_h+1)*cell_h, i_w*cell_w:(i_w+1)*cell_w]
                    region_ang = ang[i_h*cell_h:(i_h+1)*cell_h, i_w*cell_w:(i_w+1)*cell_w]
                    
                    # Weight angles by magnitude
                    hist, _ = np.histogram(region_ang, bins=bins, range=(0, np.pi), weights=region_mag)
                    if np.sum(hist) > 0:
                        features[i, idx:idx+bins] = hist / np.sum(hist)
                    idx += bins
            
            # 7. Frequency domain features (20)
            # DCT coefficients in zigzag pattern (focusing on lower frequencies)
            dct = cv2.dct(gray)
            zigzag_indices = [(0,0), (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2), (2,1), (3,0),
                            (4,0), (3,1), (2,2), (1,3), (0,4), (0,5), (1,4), (2,3), (3,2), (4,1)]
            
            for j, (y, x) in enumerate(zigzag_indices):
                if y < dct.shape[0] and x < dct.shape[1]:
                    features[i, 108+j] = dct[y, x]
                else:
                    features[i, 108+j] = 0
        
        # Feature normalization with improved robustness
        for j in range(features.shape[1]):
            feature_std = np.std(features[:, j])
            if feature_std > 1e-10:  # Avoid division by very small numbers
                features[:, j] = (features[:, j] - np.mean(features[:, j])) / feature_std
        
        return features
    
    def predict_image(self, image_data):
        """Process image data with enhanced preprocessing for better prediction"""
        try:
            # Convert image data to numpy array
            image = Image.open(BytesIO(image_data))
            
            # Enhanced preprocessing pipeline
            # 1. Resize with better quality
            image = image.resize((32, 32), Image.LANCZOS)
            img_array = np.array(image)
            
            # 2. Better handling of different color formats
            if len(img_array.shape) == 2:  # Grayscale
                img_array = np.stack([img_array]*3, axis=2)
            elif img_array.shape[2] == 4:  # RGBA
                # Better alpha handling - blend with white background
                alpha = img_array[:,:,3:4] / 255.0
                rgb = img_array[:,:,:3]
                white_bg = np.ones_like(rgb) * 255
                img_array = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                img_array = img_array[:,:,:3]
                
            # 3. Normalize image
            img_array = img_array.astype(np.float32) / 255.0
            
            # 4. Apply simple contrast normalization
            for c in range(3):
                channel = img_array[:,:,c]
                p5 = np.percentile(channel, 5)
                p95 = np.percentile(channel, 95)
                if p95 > p5:
                    img_array[:,:,c] = np.clip((channel - p5) / (p95 - p5), 0, 1)
            
            # 5. Extract features and apply feature selection
            features = self.extract_enhanced_features(img_array)
            if self.selected_indices is not None:
                features = features[:, self.selected_indices]
                
            # 6. Get prediction and probabilities
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # 7. Enhanced confidence estimation
            confidence = float(probabilities[prediction])
            
            # 8. Prepare detailed response
            result = {
                'predicted_class': self.class_names[prediction],
                'confidence': confidence,
                'class_probabilities': {self.class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
            }
            
            # Add interpretability info based on feature importance
            if hasattr(self.model, 'feature_importances'):
                # Get top 5 most important features for this prediction
                top_feature_indices = np.argsort(self.model.feature_importances)[-5:][::-1]
                top_features = {
                    f"feature_{idx}": float(self.model.feature_importances[idx])
                    for idx in top_feature_indices
                }
                result['top_features'] = top_features
            
            return result
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return {'error': str(e)}
    
    def handle_client(self, client_socket, address):
        """Handle individual client connection"""
        print(f"Connected to client: {address}")
        
        try:
            # Receive message header containing content length
            header = client_socket.recv(8)
            if not header:
                return
                
            content_length = int.from_bytes(header, byteorder='big')
            
            # Receive the full message content
            chunks = []
            bytes_received = 0
            
            while bytes_received < content_length:
                chunk = client_socket.recv(min(4096, content_length - bytes_received))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_received += len(chunk)
                
            data = b''.join(chunks)
            
            # Deserialize data
            request = pickle.loads(data)
            
            # Process request
            response = {'status': 'error', 'message': 'Unknown request type'}
            
            if 'type' in request:
                if request['type'] == 'image':
                    print("Processing image prediction request")
                    # Process image
                    prediction = self.predict_image(request['image_data'])
                    
                    response = {
                        'status': 'success',
                        'prediction': prediction
                    }
            
            # Serialize and send response
            response_data = pickle.dumps(response)
            response_length = len(response_data).to_bytes(8, byteorder='big')
            
            client_socket.sendall(response_length)
            client_socket.sendall(response_data)
            
        except Exception as e:
            print(f"Error handling client {address}: {str(e)}")
        finally:
            client_socket.close()
            print(f"Connection closed with {address}")
    
    def start(self):
        """Start the server"""
        # Load the model first
        try:
            self.load_model()
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return False
            
        # Create the server socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.running = True
            print(f"Server started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    client_thread = threading.Thread(target=self.handle_client, args=(client_socket, address))
                    client_thread.daemon = True
                    client_thread.start()
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {str(e)}")
                    
        except Exception as e:
            print(f"Server error: {str(e)}")
        finally:
            self.stop()
            
        return True
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            print("Server stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Forest Image Classification Server')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=9000, help='Server port')
    parser.add_argument('--model', type=str, default='animal_rf_model.npy', 
                        help='Path to model file')
    parser.add_argument('--train_dir', type=str, default='train', 
                        help='Directory containing training data')
    parser.add_argument('--test_dir', type=str, default='test', 
                        help='Directory containing test data')
    
    args = parser.parse_args()
    
    server = RFServer(host=args.host, port=args.port, model_path=args.model,
                     train_dir=args.train_dir, test_dir=args.test_dir)
    server.start()
