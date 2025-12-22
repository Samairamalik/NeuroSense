import cv2
import numpy as np
import joblib
import os

def preprocess_mri(image):
    """Preprocess MRI image for prediction"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0   # normalization (important)
    return image.flatten()

def predict_image(image_path, model_path):
    """Predict Parkinson's from MRI scan"""
    model = joblib.load(model_path)

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    features = preprocess_mri(image).reshape(1, -1)

    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    parkinson_prob = probs[1]  # Parkinson class probability
    normal_prob = probs[0]     # Normal class probability

    label = "parkinson" if pred == 1 else "normal"
    return label, parkinson_prob, normal_prob

def test_single_mri(image_path, model_path):
    """Test a single MRI scan with detailed output"""
    
    print("\n" + "=" * 60)
    print("MRI SCAN ANALYSIS")
    print("=" * 60)
    
    try:
        label, parkinson_prob, normal_prob = predict_image(image_path, model_path)
        
        print(f"Image: {image_path}")
        print(f"Model: {model_path}")
        print("\n" + "-" * 60)
        print("PREDICTION RESULTS")
        print("-" * 60)
        print(f"Prediction: {label.upper()}")
        print(f"\nProbabilities:")
        print(f"  • Parkinson's: {parkinson_prob*100:.2f}%")
        print(f"  • Normal:      {normal_prob*100:.2f}%")
        
        print("\n" + "-" * 60)
        print("FINAL ASSESSMENT")
        print("-" * 60)
        
        if parkinson_prob > 0.5:
            confidence = parkinson_prob * 100
            print(f"⚠️  Conclusion: Likely Parkinson's Disease")
            print(f"   Confidence: {confidence:.2f}%")
        else:
            confidence = normal_prob * 100
            print(f"✓ Conclusion: Likely Normal")
            print(f"   Confidence: {confidence:.2f}%")
        
        print("=" * 60)
        
        return label, parkinson_prob, normal_prob
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def test_multiple_mri(image_paths, model_path):
    """Test multiple MRI scans"""
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE MRI SCANS")
    print("=" * 60)
    
    all_parkinson_probs = []
    all_normal_probs = []
    all_labels = []
    
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n[{i}/{len(image_paths)}] {image_path}")
        print("-" * 40)
        
        try:
            label, parkinson_prob, normal_prob = predict_image(image_path, model_path)
            
            all_labels.append(label)
            all_parkinson_probs.append(parkinson_prob)
            all_normal_probs.append(normal_prob)
            
            print(f"Prediction: {label.upper()}")
            print(f"Parkinson's: {parkinson_prob*100:.2f}%")
            print(f"Normal:      {normal_prob*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    # Calculate averages
    if all_parkinson_probs:
        avg_parkinson = np.mean(all_parkinson_probs)
        avg_normal = np.mean(all_normal_probs)
        
        print("\n" + "=" * 60)
        print("AGGREGATED RESULTS")
        print("=" * 60)
        print(f"Scans analyzed: {len(all_parkinson_probs)}")
        print(f"\nAverage Probabilities:")
        print(f"  • Parkinson's: {avg_parkinson*100:.2f}%")
        print(f"  • Normal:      {avg_normal*100:.2f}%")
        print(f"\nIndividual predictions: {all_labels}")
        
        print("\n" + "-" * 60)
        print("FINAL ASSESSMENT")
        print("-" * 60)
        
        if avg_parkinson > 0.5:
            print(f"⚠️  Conclusion: Likely Parkinson's Disease")
            print(f"   Average confidence: {avg_parkinson*100:.2f}%")
        else:
            print(f"✓ Conclusion: Likely Normal")
            print(f"   Average confidence: {avg_normal*100:.2f}%")
        
        print("=" * 60)
        
        return all_labels, avg_parkinson, avg_normal
    else:
        print("\n❌ No valid predictions made")
        return None, None, None

def batch_test_folder(folder_path, model_path):
    """Test all MRI images in a folder"""
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(folder_path, file))
    
    if not image_files:
        print(f"❌ No image files found in {folder_path}")
        return
    
    print(f"\nFound {len(image_files)} MRI scans in folder: {folder_path}")
    test_multiple_mri(image_files, model_path)

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    
    import sys
    
    # Configuration
    MODEL_PATH = "models/parkinsons_mri_model.pkl"
    IMAGE_PATH = "mri/sample.png"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image file not found at {IMAGE_PATH}")
        sys.exit(1)
    
    # Test single MRI
    test_single_mri(IMAGE_PATH, MODEL_PATH)
    
    print("\n\n")
    
    # Optional: Test multiple MRI scans
    # Uncomment to test multiple images
    """
    print("Testing Multiple MRI Scans")
    mri_images = [
        "mri/sample1.png",
        "mri/sample2.png",
        "mri/sample3.png",
    ]
    test_multiple_mri(mri_images, MODEL_PATH)
    """
    
    # Optional: Test entire folder
    # Uncomment to test all images in a folder
    """
    batch_test_folder("mri/test_scans", MODEL_PATH)
    """