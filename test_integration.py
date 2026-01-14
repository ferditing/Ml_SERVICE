#!/usr/bin/env python3
"""
Integration test for ML Service + Backend Worker
Tests both the ML service directly and simulates backend requests
"""

import requests
import json
import time

ML_SERVICE_URL = "http://localhost:8001"

def test_health_check():
    """Test 1: Health check endpoint"""
    print("\nTest 1: Health Check")
    print("-" * 50)
    try:
        resp = requests.get(f"{ML_SERVICE_URL}/health", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"Status: {resp.status_code}")
        print(f"Features count: {data['features_count']}")
        print(f"Labels: {data['labels']}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_simple_prediction():
    """Test 2: Simple prediction with minimal data"""
    print("\nTest 2: Simple Prediction (Minimal Data)")
    print("-" * 50)
    payload = {
        "animal_type": "cow",
        "age": 5,
        "body_temperature": 38.5,
        "symptoms": []
    }
    try:
        resp = requests.post(f"{ML_SERVICE_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print(f"Prediction: {data['predicted_label']}")
        print(f"Confidence: {data['confidence']}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_prediction_with_symptoms():
    """Test 3: Prediction with symptoms"""
    print("\nTest 3: Prediction (With Symptoms)")
    print("-" * 50)
    payload = {
        "animal_type": "goat",
        "age": 3,
        "body_temperature": 39.2,
        "symptoms": ["fever", "cough", "lethargy"]
    }
    try:
        resp = requests.post(f"{ML_SERVICE_URL}/predict", json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print(f"Prediction: {data['predicted_label']}")
        print(f"Confidence: {data['confidence']}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_different_animal_types():
    """Test 4: Test multiple animal types"""
    print("\nTest 4: Different Animal Types")
    print("-" * 50)
    animals = ["cow", "goat", "sheep", "pig"]
    results = {}
    
    for animal in animals:
        payload = {
            "animal_type": animal,
            "age": 4,
            "body_temperature": 38.8,
            "symptoms": ["loss of appetite"]
        }
        try:
            resp = requests.post(f"{ML_SERVICE_URL}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            results[animal] = {
                "label": data['predicted_label'],
                "confidence": data['confidence']
            }
            print(f"  {animal:10} -> {data['predicted_label']:20} (conf: {data['confidence']})")
        except Exception as e:
            print(f"  {animal:10} -> ERROR: {e}")
            results[animal] = {"error": str(e)}
    
    return len([r for r in results.values() if "error" not in r]) == len(animals)

def test_edge_cases():
    """Test 5: Edge cases"""
    print("\nTest 5: Edge Cases")
    print("-" * 50)
    
    test_cases = [
        {
            "name": "Very high temperature",
            "payload": {"animal_type": "cow", "age": 2, "body_temperature": 41.0, "symptoms": ["fever"]}
        },
        {
            "name": "Very low temperature",
            "payload": {"animal_type": "sheep", "age": 1, "body_temperature": 35.0, "symptoms": []}
        },
        {
            "name": "Many symptoms",
            "payload": {
                "animal_type": "pig",
                "age": 6,
                "body_temperature": 39.5,
                "symptoms": ["fever", "cough", "lethargy", "loss of appetite", "diarrhea"]
            }
        },
    ]
    
    passed = 0
    for test in test_cases:
        try:
            resp = requests.post(f"{ML_SERVICE_URL}/predict", json=test["payload"], timeout=5)
            resp.raise_for_status()
            data = resp.json()
            print(f"  {test['name']:25} -> {data['predicted_label']:20} (conf: {data['confidence']})")
            passed += 1
        except Exception as e:
            print(f"  {test['name']:25} -> ERROR: {e}")
    
    return passed == len(test_cases)

def main():
    print("\n" + "="*50)
    print("ML SERVICE INTEGRATION TEST")
    print("="*50)
    print(f"\nTarget: {ML_SERVICE_URL}")
    print("Make sure the ML service is running!")
    
    tests = [
        ("Health Check", test_health_check),
        ("Simple Prediction", test_simple_prediction),
        ("Prediction with Symptoms", test_prediction_with_symptoms),
        ("Different Animal Types", test_different_animal_types),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\nUNEXPECTED ERROR in {name}: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status:4} | {name}")
    
    total_passed = sum(1 for p in results.values() if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
