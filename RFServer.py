import socket
import numpy as np
import pickle
import threading
import os
import base64
from PIL import Image
import cv2
from io import BytesIO
from Random_Forest import RandomForestClassifier
from DatabaseHandler import DatabaseHandler
from SecurityUtils import SecurityUtils

class RFServer:
    """
    Server for Random Forest image classifier.
    Hosts the AI model and processes client requests.
    """
    
    def __init__(self, host='0.0.0.0', port=9000, model_path='vehicle_rf_model.npy', 
             train_dir='Model_pics\\train', test_dir='Model_pics\\test', training_sample_limit=10000):
        """Initialize the server with host, port, model path and data directories"""
        self.host = host
        self.port = port
        self.model_path = model_path
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.server_socket = None
        self.model = None
        self.selected_indices = None
        self.class_names = ['Cars', 'Motorbikes', 'Planes', 'Ships']
        self.running = False
        self.training_sample_limit = training_sample_limit
        # Initialize database handler
        self.db_handler = DatabaseHandler()
        
        # Security setup
        self.private_key = None
        self.public_key = None
        self.setup_security()

    def setup_security(self):
        """Set up encryption keys for secure communication"""
        try:
            # Generate RSA key pair for secure initial communication
            private_key_pem, public_key_pem = SecurityUtils.generate_rsa_key_pair()
            self.private_key = private_key_pem
            self.public_key = public_key_pem
            print("Security keys generated successfully")
        except Exception as e:
            print(f"Failed to set up security: {str(e)}")
            # Continue without encryption if key generation fails
            self.private_key = None
            self.public_key = None

    def load_model(self):
        """Load the Random Forest model from disk or train a new one, with added debugging"""
        print("Loading model...")
        if os.path.exists(self.model_path):
            try:
                model_data = np.load(self.model_path, allow_pickle=True).item()
                
                # Debug model data
                print(f"Model data loaded from {self.model_path}")
                print(f"Keys in model data: {list(model_data.keys())}")
                
                # Check if required components are present
                if 'base_learner_list' not in model_data or 'selected_indices' not in model_data:
                    print("Error: Model data is missing required components!")
                    print("Re-training model...")
                    self.train_model(
                        sample_limit=self.training_sample_limit,
                        augmentation_factor=0.3,
                        feature_count=80
                    )
                    return
                
                self.model = RandomForestClassifier()
                self.model.base_learner_list = model_data['base_learner_list']
                print(f"Loaded {len(self.model.base_learner_list)} base learners")
                
                self.model.feature_importances = model_data['feature_importances']
                self.selected_indices = model_data['selected_indices']
                print(f"Selected {len(self.selected_indices)} features")
                
                print(f"Model loaded successfully with accuracy: {model_data.get('test_accuracy', 'N/A')}")
                
                # Debug class names
                print(f"Current class names: {self.class_names}")
                if 'class_accuracies' in model_data:
                    print("Per-class accuracies:")
                    for cls, acc in model_data['class_accuracies'].items():
                        print(f"  {cls}: {acc:.4f}")
                
                
            except Exception as e:
                import traceback
                print(f"Error loading model: {str(e)}")
                print(traceback.format_exc())
                print("Re-training model...")
                self.train_model(
                    sample_limit=self.training_sample_limit,
                    augmentation_factor=0.3,
                    feature_count=80
                )
        else:
            print(f"Model not found at {self.model_path}. Starting model training...")
            self.train_model(
                sample_limit=self.training_sample_limit,
                augmentation_factor=0.3,
                feature_count=80
            )
    
    def augment_training_data(self, features, labels, augmentation_factor=0.5, max_augmented_samples=1000):
        """
        Augment training data with memory constraints
        
        Args:
            features: Feature array
            labels: Label array
            augmentation_factor: Float between 0-1 determining what fraction of original data size to add
            max_augmented_samples: Maximum number of augmented samples to generate regardless of factor
            
        Returns:
            augmented_features, augmented_labels
        """
        n_samples, n_features = features.shape
        n_augmented = min(int(n_samples * augmentation_factor), max_augmented_samples)
        
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
            
            # Create a perturbed version - use simpler perturbation to save memory
            perturbed_feature = original_feature.copy()
            
           
            noise_factor = 0.05
            noise = np.random.normal(0, noise_factor, n_features)
            perturbed_feature += noise
            
            # Store augmented example
            augmented_features[n_samples + i] = perturbed_feature
            augmented_labels[n_samples + i] = original_label
        
        return augmented_features, augmented_labels

    def train_model(self, sample_limit=None, augmentation_factor=0.5, feature_count=80):
        """Train a Random Forest model with memory constraints
        
        Args:
            sample_limit: Maximum number of samples to use for training (None=use all)
            augmentation_factor: Factor to use for data augmentation (0=no augmentation)
            feature_count: Number of features to select
        """
        
        # Load and preprocess the data
        print("Loading and preprocessing training data...")
        X_train, y_train, X_test, y_test = self.load_custom_dataset(self.train_dir, self.test_dir)
        
        # Limit training samples if requested
        if sample_limit and sample_limit < len(X_train):
            print(f"Limiting training samples to {sample_limit}")
            indices = np.random.choice(len(X_train), sample_limit, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        # Extract features using the specialized vehicle feature extractor
        print("Extracting vehicle-specific features from training images...")
        train_features = self.extract_vehicle_features(X_train)
        test_features = self.extract_vehicle_features(X_test)
        
        # Feature selection - use more features for vehicle classification
        print("Performing feature selection...")
        self.selected_indices = self.select_features(train_features, y_train, n_features=feature_count)
        
        # Apply feature selection
        train_features_selected = train_features[:, self.selected_indices]
        test_features_selected = test_features[:, self.selected_indices]
        print(test_features_selected[0])
        # Free memory
        del train_features, test_features
        
        # Data augmentation for training (if factor > 0)
        if augmentation_factor > 0:
            print("Performing data augmentation...")
            augmented_features, augmented_labels = self.augment_training_data(
                train_features_selected, y_train, 
                augmentation_factor=augmentation_factor
            )
        else:
            augmented_features = train_features_selected
            augmented_labels = y_train
        
        # Train the model with optimized parameters for vehicle classification
        print("Training Random Forest model with optimized parameters...")
        self.model = RandomForestClassifier(
            n_base_learner=100,  # Scale based on available data
            max_depth=15,                   # Reduced from 18
            min_samples_leaf=3,
            min_information_gain=0.001,
            numb_of_features_splitting='sqrt',
            bootstrap_sample_size=None
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
        for class_idx in range(len(self.class_names)):
            class_mask = (y_test == class_idx)
            if np.sum(class_mask) > 0.0:
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
        
        # except Exception as e:
        #     print(f"Error during model training: {str(e)}")
        #     raise RuntimeError(f"Failed to train model: {str(e)}")
    
    def validate_test_images(self):
        """
        Validates the first 10 motorcycle images from the test data
        and shows classification results for each one.
        """
        import os
        from PIL import Image
        
        # Path to the motorcycle test images
        motorbike_test_dir = os.path.join('Model_pics', 'test', 'Motorbikes')
        
        if not os.path.exists(motorbike_test_dir):
            print(f"Error: Motorbike test directory not found at {motorbike_test_dir}")
            return
        
        # Get list of image files
        image_files = [f for f in os.listdir(motorbike_test_dir) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        
        # Limit to first 10 images
        image_files = image_files[:10]
        
        if not image_files:
            print("No motorbike test images found")
            return
        
        print(f"Validating {len(image_files)} motorbike test images...")
        
        # Process each image
        for i, img_file in enumerate(image_files):
            try:
                # Load and preprocess the image
                img_path = os.path.join(motorbike_test_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((32, 32), Image.LANCZOS)  # Resize to 32x32
                img_array = np.array(img) / 255.0  # Normalize to [0,1]
                
                # Add a batch dimension
                img_array = np.expand_dims(img_array, axis=0)
                
                # Extract features
                features = self.extract_vehicle_features(img_array)
                features_selected = features[:, self.selected_indices]
                
                # Get predictions
                predictions = self.model.predict(features_selected)
                pred_probs = self.model.predict_proba(features_selected)
                
                # Get predicted class
                pred_class_idx = predictions[0]
                pred_class_name = self.class_names[pred_class_idx]
                
                # Calculate confidence
                confidence = pred_probs[0][pred_class_idx]
                
                # Display results
                print(f"\nImage {i+1}: {img_file}")
                print(f"Predicted class: {pred_class_name}")
                print(f"Confidence: {confidence:.4f}")
                
                # Display all class probabilities
                print("Class probabilities:")
                for class_idx, class_name in enumerate(self.class_names):
                    prob = pred_probs[0][class_idx]
                    print(f"  - {class_name}: {prob:.4f}")
                
                # Check if classification is correct (should be motorbike = index 1)
                is_correct = pred_class_idx == 1  # 1 is the index for Motorbikes
                print(f"Classification {'correct' if is_correct else 'incorrect'}")
                
            except Exception as e:
                print(f"Error processing image {img_file}: {str(e)}")
        
        # Overall statistics
        print("\nFinished validation of motorbike test images")

    def load_custom_dataset(self, train_dir, test_dir, max_samples_per_class=3000):
        """Load vehicle classes from custom image directories"""
        import os
        from PIL import Image
        import numpy as np
        

        # Path to the Model_pics directory
        base_dir = 'Model_pics'
        train_path = os.path.join(base_dir, 'train')
        test_path = os.path.join(base_dir, 'test')
        
        print(f"Loading custom dataset from {base_dir}...")
        
        # Check if directories exist
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            raise FileNotFoundError(f"Could not find train/test directories in {base_dir}")
        
        # Lists to store images and labels
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        # Count samples per class to enforce limits
        train_class_counts = {cls: 0 for cls in self.class_names}
        test_class_counts = {cls: 0 for cls in self.class_names}
        
        # Load training images
        print("Loading training images...")
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(train_path, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found")
                continue
            
            
            for img_file in os.listdir(class_dir):
                # Check if we've reached the sample limit for this class
                if train_class_counts[class_name] >= max_samples_per_class:
                    break
                
                   
                # Load only image files
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        img_path = os.path.join(class_dir, img_file)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((32, 32), Image.LANCZOS)  # Resize to 32x32
                        img_array = np.array(img) / 255.0  # Normalize to [0,1]
                        
                        X_train.append(img_array)
                        y_train.append(class_idx)
                        train_class_counts[class_name] += 1
                    except Exception as e:
                        print(f"Error loading image {img_path}: {str(e)}")
        
        # Load test images
        print("Loading test images...")
        max_test_samples = max_samples_per_class // 5  # Use fewer test samples
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(test_path, class_name)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Class directory {class_dir} not found")
                continue

            for img_file in os.listdir(class_dir):
                # Check if we've reached the sample limit for this class
                if test_class_counts[class_name] >= max_test_samples:
                    break
                    
                # Load only image files
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    try:
                        img_path = os.path.join(class_dir, img_file)
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((32, 32), Image.LANCZOS)  # Resize to 32x32
                        img_array = np.array(img) / 255.0  # Normalize to [0,1]
                        
                        X_test.append(img_array)
                        y_test.append(class_idx)
                        test_class_counts[class_name] += 1
                    except Exception as e:
                        print(f"Error loading image {img_path}: {str(e)}")
        
        # Convert to numpy arrays
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int32)
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int32)
        
        # Print dataset statistics
        print(f"Dataset prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        for class_idx, class_name in enumerate(self.class_names):
            train_count = np.sum(y_train == class_idx)
            test_count = np.sum(y_test == class_idx)
            print(f"  {class_name}: {train_count} training, {test_count} test images")
        
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
    
    def extract_vehicle_features(self, images):
        """
        Extract more robust features for vehicle classification.
        Returns consistent features for an image regardless of batch size.

        Args:
            images: A numpy array of images. Can be a single image [H, W, C]
                    or multiple images [N, H, W, C]. Assumes images are float type
                    in the range [0, 1] and RGB format.

        Returns:
            A numpy array of features [N, feature_dim], where N is the number
            of images and feature_dim is the number of extracted features (120).
            Features are NOT normalized across the batch in this version.
        """
        # Make sure images is a 4D array: [n_samples, height, width, channels]
        if len(images.shape) == 3:  # Single image with shape [height, width, channels]
            images = np.expand_dims(images, axis=0)
        elif len(images.shape) != 4:
             raise ValueError("Input images must be 3D [H, W, C] or 4D [N, H, W, C]")

        n_samples = images.shape[0]
        # Define feature dimension clearly
        feature_dim = 120
        features = np.zeros((n_samples, feature_dim), dtype=np.float32)

        # Define batch size (can be adjusted based on memory)
        batch_size = 50

        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_images = images[batch_start:batch_end]

            for i, img in enumerate(batch_images):
                global_i = batch_start + i # Index in the main features array

                # --- Preprocessing ---
                # Ensure image is float [0, 1] for internal calculations
                # but create uint8 version for OpenCV functions needing it
                if img.dtype != np.uint8:
                    img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                else:
                    # If already uint8, convert to float for calculations
                    img_uint8 = img
                    img = img.astype(np.float32) / 255.0

                # Create a copy for normalization to avoid modifying the input array slice
                img_norm = img.copy()

                # Enhanced contrast normalization (applied per image)
                for c in range(3):
                    channel = img_norm[:, :, c]
                    p2, p98 = np.percentile(channel, [2, 98])
                    if p98 > p2:
                        img_norm[:, :, c] = np.clip((channel - p2) / (p98 - p2), 0, 1)
                    elif p98 == p2 and p2 > 0: # Handle flat channels
                         img_norm[:, :, c] = (channel >= p2).astype(float)
                    # else: channel is likely all zero, leave it


                # Convert normalized image back to uint8 for certain cv2 functions
                img_norm_uint8 = (img_norm * 255).astype(np.uint8)

                # Convert to different color spaces using the contrast-normalized uint8 image
                # Convert results back to float [0, 1] for consistency
                gray = cv2.cvtColor(img_norm_uint8, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
                hsv = cv2.cvtColor(img_norm_uint8, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
                gray_uint8 = (gray * 255).astype(np.uint8) # uint8 gray for Canny/contours


                # --- Feature Calculation ---
                current_feature_idx = 0

                # 1. Color Histograms (RGB + HSV, 5 bins each = 30 features)
                # Use the contrast-normalized image (img_norm)
                hist_bins = 5
                num_color_features = 2 * 3 * hist_bins # RGB + HSV, 3 channels, 5 bins
                color_hist_features = np.zeros(num_color_features, dtype=np.float32)
                temp_idx = 0
                # RGB
                for c in range(3):
                    hist, _ = np.histogram(img_norm[:, :, c], bins=hist_bins, range=(0, 1))
                    sum_hist = np.sum(hist)
                    if sum_hist > 0:
                         color_hist_features[temp_idx : temp_idx + hist_bins] = hist / sum_hist
                    temp_idx += hist_bins
                # HSV
                for c in range(3):
                    hist, _ = np.histogram(hsv[:, :, c], bins=hist_bins, range=(0, 1))
                    sum_hist = np.sum(hist)
                    if sum_hist > 0:
                         color_hist_features[temp_idx : temp_idx + hist_bins] = hist / sum_hist
                    temp_idx += hist_bins
                features[global_i, current_feature_idx : current_feature_idx + num_color_features] = color_hist_features
                current_feature_idx += num_color_features # current_feature_idx = 30

                # 2. Color Statistics (Mean, Std, Skewness for RGB = 9 features)
                # Use the contrast-normalized image (img_norm)
                num_stat_features = 3 * 3 # 3 channels, 3 stats
                stat_features = np.zeros(num_stat_features, dtype=np.float32)
                temp_idx = 0
                for c in range(3):
                    channel = img_norm[:, :, c].ravel() # Flatten for stats
                    mean_val = np.mean(channel)
                    std_val = np.std(channel)
                    stat_features[temp_idx] = mean_val
                    stat_features[temp_idx + 1] = std_val
                    # Avoid division by zero in skewness calculation
                    if std_val > 1e-6:
                        skewness = np.mean(((channel - mean_val) / std_val)**3)
                        stat_features[temp_idx + 2] = skewness
                    else:
                        stat_features[temp_idx + 2] = 0.0
                    temp_idx += 3
                features[global_i, current_feature_idx : current_feature_idx + num_stat_features] = stat_features
                current_feature_idx += num_stat_features # current_feature_idx = 39

                # 3. Shape Features (Edges)
                # 3a. Multi-scale Canny edge density in 4 quadrants (3 scales * 4 quadrants = 12 features)
                num_edge_density_features = 3 * 4
                edge_density_features = np.zeros(num_edge_density_features, dtype=np.float32)
                temp_idx = 0
                h, w = gray_uint8.shape
                canny_thresholds = [(20, 100), (50, 150), (80, 200)]
                for low, high in canny_thresholds:
                    edges = cv2.Canny(gray_uint8, low, high)
                    # Calculate density (mean value) - ensure division is safe
                    h_half, w_half = h // 2, w // 2
                    q1_size = (h_half * w_half) if (h_half * w_half) > 0 else 1
                    q2_size = (h_half * (w - w_half)) if (h_half * (w - w_half)) > 0 else 1
                    q3_size = ((h - h_half) * w_half) if ((h - h_half) * w_half) > 0 else 1
                    q4_size = ((h - h_half) * (w - w_half)) if ((h - h_half) * (w - w_half)) > 0 else 1

                    edge_density_features[temp_idx]     = np.sum(edges[:h_half, :w_half]) / q1_size / 255.0
                    edge_density_features[temp_idx + 1] = np.sum(edges[:h_half, w_half:]) / q2_size / 255.0
                    edge_density_features[temp_idx + 2] = np.sum(edges[h_half:, :w_half]) / q3_size / 255.0
                    edge_density_features[temp_idx + 3] = np.sum(edges[h_half:, w_half:]) / q4_size / 255.0
                    temp_idx += 4
                features[global_i, current_feature_idx : current_feature_idx + num_edge_density_features] = edge_density_features
                current_feature_idx += num_edge_density_features # current_feature_idx = 51

                # 3b. Edge Orientation Histogram (8 bins = 8 features)
                num_edge_orient_features = 8
                edge_orient_features = np.zeros(num_edge_orient_features, dtype=np.float32)
                # Use Sobel on the float grayscale image
                gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False) # Angle in radians [0, 2pi]
                # Use a single Canny result for the mask
                edges_mask = cv2.Canny(gray_uint8, 50, 150) > 0

                edge_points = np.where(edges_mask)
                if len(edge_points[0]) > 0:  # Check if there are any edge points
                    # Use angles only where edges exist, range [0, pi]
                    edge_angles = ang[edge_points] % np.pi
                    hist_edges, _ = np.histogram(edge_angles, bins=num_edge_orient_features, range=(0, np.pi))
                    sum_hist = np.sum(hist_edges)
                    if sum_hist > 0:
                        edge_orient_features = hist_edges / sum_hist
                features[global_i, current_feature_idx : current_feature_idx + num_edge_orient_features] = edge_orient_features
                current_feature_idx += num_edge_orient_features # current_feature_idx = 59

                # 4. Texture Features (Gabor filter responses: 4 orientations * 2 sigmas * 2 stats = 16 features)
                num_gabor_features = 4 * 2 * 2
                gabor_features = np.zeros(num_gabor_features, dtype=np.float32)
                temp_idx = 0
                gabor_params = {'ksize': (5, 5), 'gamma': 0.5, 'psi': 0, 'ktype': cv2.CV_32F}
                for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    for sigma in [1.0, 3.0]:
                        kernel = cv2.getGaborKernel(**gabor_params, sigma=sigma, theta=theta, lambd=10.0)
                        # Filter the float grayscale image
                        filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                        gabor_features[temp_idx] = np.mean(filtered)
                        gabor_features[temp_idx + 1] = np.std(filtered)
                        temp_idx += 2
                features[global_i, current_feature_idx : current_feature_idx + num_gabor_features] = gabor_features
                current_feature_idx += num_gabor_features # current_feature_idx = 75

                # 5. HOG-like Features (Simplified: Mean gradient magnitude per cell = 45 features approx)
                # Note: This is a simplified version. A full HOG implementation is more complex.
                # We aim for 120 total features. Current = 75. Need 45 more.
                num_hog_features_target = feature_dim - current_feature_idx # Target 45 features
                hog_features = np.zeros(num_hog_features_target, dtype=np.float32)

                h, w = gray.shape
                # Adjust cell size dynamically to get roughly the desired number of features
                # Aim for roughly sqrt(45) cells ~ 6x7 or 7x7 grid
                approx_grid_size = int(np.ceil(np.sqrt(num_hog_features_target)))
                cell_size_y = max(1, h // approx_grid_size)
                cell_size_x = max(1, w // approx_grid_size)
                n_cells_y = h // cell_size_y
                n_cells_x = w // cell_size_x

                temp_idx = 0
                # Use magnitude 'mag' calculated earlier from Sobel
                for y in range(n_cells_y):
                    for x in range(n_cells_x):
                        if temp_idx >= num_hog_features_target: # Stop if we exceed target
                            break
                        cell_mag = mag[y*cell_size_y:(y+1)*cell_size_y, x*cell_size_x:(x+1)*cell_size_x]
                        # Use mean magnitude as the feature for this cell
                        if cell_mag.size > 0:
                             hog_features[temp_idx] = np.mean(cell_mag)
                        else:
                             hog_features[temp_idx] = 0.0
                        temp_idx += 1
                    if temp_idx >= num_hog_features_target:
                        break

                # Assign the calculated HOG features (up to the target number)
                features[global_i, current_feature_idx : current_feature_idx + temp_idx] = hog_features[:temp_idx]

                # Replace any potential NaNs or Infs with 0
                features[global_i, :] = np.nan_to_num(features[global_i, :], nan=0.0, posinf=0.0, neginf=0.0)



        return features
    
    def predict_image(self, image_data):
        """Process image data with the exact same pipeline as training/test images"""
        if self.selected_indices is None:
            print("Error: No feature indices selected. Model may not be properly loaded.")
            return {"error": "Model initialization error"}
        
        try:
            # Print shape for debugging
            print(f"Input image data shape: {image_data.shape}")
            
            # Extract features using the same function used for training
            features = self.extract_vehicle_features(image_data)
            
            # Apply feature selection - make sure to use the same indices from training
            if features.shape[1] != 120:
                print(f"Error: Expected 120 features, got {features.shape[1]}")
                return {"error": "Feature extraction error"}
                
            print(f"Selected indices shape: {self.selected_indices.shape}")
            features_selected = features[:, self.selected_indices]
            print(features_selected[0])
            # Debug info about selected features
            print(f"Selected features shape: {features_selected.shape}")
            print(f"First few feature values: {features_selected[0, :5]}")
            
            # Use the model to predict
            predictions = self.model.predict(features_selected)
            pred_probs = self.model.predict_proba(features_selected)
            
            pred_class_idx = predictions[0]
            pred_class_name = self.class_names[pred_class_idx]
            
            # Calculate confidence (probability of the predicted class)
            confidence = pred_probs[0][pred_class_idx]
            
            # Show prediction probabilities for debugging
            print("Prediction probabilities:")
            for i, prob in enumerate(pred_probs[0]):
                print(f"  Class {self.class_names[i]}: {prob:.4f}")
            
            # Create a dictionary of all class probabilities
            class_probabilities = {self.class_names[i]: float(pred_probs[0][i]) for i in range(len(self.class_names))}
            
            # Create the result with more detailed information
            result = {
                'predicted_class': pred_class_name,
                'confidence': float(confidence),
                'class_probabilities': class_probabilities
            }
            
            return result
        except Exception as e:
            import traceback
            print(f"Error in predict_image: {str(e)}")
            print(traceback.format_exc())
            return {"error": str(e)}
            
    def handle_client(self, client_socket, address):
        """Handle individual client connection continuously"""
        print(f"Connected to client: {address}")
        client_public_key = None
        session_key = None
        session_iv = None
        authenticated_user_id = None

        try:
            # Initial security handshake (RSA part remains the same)
            if self.public_key is not None:
                # Send server's public key first
                # Note: Handshake messages are sent unencrypted initially
                self.send_message(client_socket, {'type': 'public_key', 'key': self.public_key.decode('utf-8')})

                # Receive client's public key
                # Note: Handshake messages received unencrypted initially
                client_key_data = self.receive_message(client_socket)
                if client_key_data and 'type' in client_key_data and client_key_data['type'] == 'public_key':
                    client_public_key = client_key_data['key'].encode('utf-8')
                    print(f"Received public key from client {address}")

                    # Generate AES session key and IV
                    session_key, session_iv = SecurityUtils.generate_aes_key()

                    # Encrypt session key/IV with client's public RSA key
                    # Base64 encode the raw bytes *before* RSA encryption
                    key_b64 = base64.b64encode(session_key).decode('utf-8')
                    iv_b64 = base64.b64encode(session_iv).decode('utf-8')

                    encrypted_key_b64 = SecurityUtils.encrypt_with_public_key(
                        key_b64,
                        client_public_key
                    )
                    encrypted_iv_b64 = SecurityUtils.encrypt_with_public_key(
                        iv_b64,
                        client_public_key
                    )

                    # Send encrypted session key/IV (unencrypted wrapper)
                    self.send_message(client_socket, {
                        'type': 'session_key',
                        'key': encrypted_key_b64, # Send base64 encoded encrypted string
                        'iv': encrypted_iv_b64   # Send base64 encoded encrypted string
                    })

                    print(f"Sent AES session key to client {address}")
                else:
                    print(f"Invalid key exchange with client {address}. Closing connection.")
                    client_socket.close()
                    return # Exit handler thread
            else:
                print("Server RSA keys not set up. Communication will be unencrypted.")
                # Decide if unencrypted communication is allowed or close connection

            # Main request handling loop (now uses AES session key)
            while True:
                # Receive potentially encrypted message using the established session key
                request = self.receive_message(client_socket, session_key, session_iv)
                if request is None: # Indicates disconnection or error in receive_message
                    print(f"Client {address} disconnected or sent invalid request.")
                    break

                # Process request based on type (logic remains the same)
                response = {'status': 'error', 'message': 'Unknown request type'}

                if 'type' in request:
                    req_type = request['type']
                    print(f"Received request type: {req_type} from {address}") # Debugging

                    # Authentication requests
                    if req_type == 'register':
                         # ... (registration logic) ...
                         success, message = self.db_handler.register_user(
                             request['username'],
                             request['password'],
                             request.get('email')
                         )
                         response = {
                             'status': 'success' if success else 'error',
                             'message': message
                         }

                    elif req_type == 'login':
                         # ... (login logic) ...
                         success, result = self.db_handler.authenticate_user(
                             request['username'],
                             request['password']
                         )
                         if success:
                             authenticated_user_id = result
                             preferred_server = self.db_handler.get_user_server_preference(authenticated_user_id)
                             response = {
                                 'status': 'success',
                                 'user_id': authenticated_user_id,
                                 'preferred_server': preferred_server
                             }
                             print(f"User {request['username']} authenticated successfully.")
                         else:
                             response = {
                                 'status': 'error',
                                 'message': result
                             }
                             print(f"Authentication failed for user {request['username']}.")

                    # Image classification requests
                    elif req_type == 'image':
                        # Requires authentication first in a real scenario
                        # if authenticated_user_id is None:
                        #    response = {'status': 'error', 'message': 'Authentication required'}
                        # else:
                        if 'image_data' in request:
                            print(f"Processing image from {address}")
                            prediction = self.predict_image(request['image_data'])
                            response = {
                                'status': 'success',
                                'prediction': prediction
                            }
                        else:
                            response['message'] = 'Missing image_data in request'
                    # Add other request types (e.g., update_preferences) here
                    else:
                         response['message'] = f"Unsupported request type: {req_type}"


                # Send potentially encrypted response using the established session key
                if not self.send_message(client_socket, response, session_key, session_iv):
                    print(f"Failed to send response to {address}. Closing connection.")
                    break # Exit loop if send fails

        except (ConnectionAbortedError, ConnectionResetError) as conn_err:
            print(f"Connection error with client {address}: {conn_err}")
        except socket.timeout:
            print(f"Socket timeout with client {address}")
        except Exception as e:
            import traceback
            print(f"Unexpected error handling client {address}: {str(e)}")
            print(traceback.format_exc())
        finally:
            print(f"Closing connection to client: {address}")
            client_socket.close()
    
    def send_message(self, client_socket, message_data, session_key=None, session_iv=None):
        """Send a message to the client, optionally encrypted with AES session key."""
        try:
            # 1. Serialize message data using pickle
            data_bytes = pickle.dumps(message_data)

            # 2. Encrypt the raw bytes if session key is available
            #    (Skip encryption for initial handshake messages where keys aren't set)
            if session_key and session_iv:
                encrypted_data = SecurityUtils.encrypt_with_aes(
                    data_bytes,
                    session_key,
                    session_iv
                )
                # print(f"Sending {len(encrypted_data)} encrypted bytes.") # Debug
            else:
                # Send unencrypted (e.g., for public key exchange)
                encrypted_data = data_bytes
                # print(f"Sending {len(encrypted_data)} unencrypted bytes.") # Debug


            # 3. Send message length followed by the (potentially encrypted) message bytes
            message_length = len(encrypted_data).to_bytes(8, byteorder='big')
            client_socket.sendall(message_length)
            client_socket.sendall(encrypted_data)
            return True # Indicate success

        except (socket.error, pickle.PicklingError, Exception) as e:
            print(f"Error sending message: {str(e)}")
            # Don't raise here, return False to allow handle_client to close socket
            return False # Indicate failure

    def receive_message(self, client_socket, session_key=None, session_iv=None):
        """Receive a message from the client, optionally decrypting with AES session key."""
        try:
            # 1. Receive response length
            header = client_socket.recv(8)
            if not header or len(header) < 8:
                print("Connection closed by client or incomplete header received.")
                return None # Indicate disconnection

            content_length = int.from_bytes(header, byteorder='big')
            if content_length == 0:
                 print("Received zero length message.")
                 return None # Or handle appropriately
            if content_length > 10 * 1024 * 1024: # Add a size limit (e.g., 10MB)
                 print(f"Message too large: {content_length} bytes. Closing connection.")
                 return None


            # 2. Receive full response bytes
            chunks = []
            bytes_received = 0
            client_socket.settimeout(20.0) # Increased timeout for potentially larger data
            while bytes_received < content_length:
                # Read in chunks to avoid large memory allocation
                chunk_size = min(4096, content_length - bytes_received)
                chunk = client_socket.recv(chunk_size)
                if not chunk:
                    print("Connection broken while receiving data.")
                    return None # Indicate disconnection
                chunks.append(chunk)
                bytes_received += len(chunk)
            client_socket.settimeout(None) # Reset timeout
            received_data = b''.join(chunks)
            # print(f"Received {len(received_data)} bytes.") # Debug

            # 3. Decrypt if session key is available
            #    (Skip decryption for initial handshake messages)
            if session_key and session_iv:
                try:
                    decrypted_data_bytes = SecurityUtils.decrypt_with_aes(
                        received_data,
                        session_key,
                        session_iv
                    )
                    # print("Message decrypted successfully.") # Debug
                except Exception as e: # Catch potential decryption/padding errors
                    print(f"Error decrypting message: {str(e)}. Invalid key or corrupted data?")
                    # Don't return None immediately, maybe log and close later
                    # For now, return None to signal error
                    return None
            else:
                 # Assume unencrypted (e.g., for public key exchange)
                 decrypted_data_bytes = received_data
                 # print("Message treated as unencrypted.") # Debug

            # 4. Deserialize response using pickle
            return pickle.loads(decrypted_data_bytes)

        except socket.timeout:
            print("Socket timeout during receive.")
            return None # Indicate timeout error
        except (socket.error, pickle.UnpicklingError, EOFError, MemoryError, Exception) as e:
            print(f"Error in receive_message: {str(e)}")
            return None

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
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=9000, help='Server port')
    parser.add_argument('--model', type=str, default='vehicle_rf_model.npy', 
                        help='Path to model file')
    parser.add_argument('--train_dir', type=str, default='train', 
                        help='Directory containing training data')
    parser.add_argument('--test_dir', type=str, default='test', 
                        help='Directory containing test data')
    parser.add_argument('--training_limit', type=int, default=10000,
                        help='Maximum number of training samples to use')
    
    args = parser.parse_args()
    
    server = RFServer(host=args.host, port=args.port, model_path=args.model,
                     train_dir=args.train_dir, test_dir=args.test_dir,
                     training_sample_limit=args.training_limit)
    server.start()