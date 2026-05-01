import os
import sys
sys.path.append(os.getcwd())
import psutil
import gc
from engine import get_models, analyze_hallucination

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def test_memory():
    print(f"Initial Memory: {get_memory_usage():.2f} MB")
    
    print("Loading models...")
    get_models()
    print(f"Memory after loading models: {get_memory_usage():.2f} MB")
    
    source = "The 2019 Cambridge Ornithology Review found that the average airspeed velocity of a European swallow is roughly 11 meters per second, or 24 miles per hour."
    text = "According to the 2019 Cambridge Ornithology Review, European swallows fly at 24 miles per hour."
    
    print("Running analysis...")
    analyze_hallucination(source, text)
    print(f"Memory after analysis: {get_memory_usage():.2f} MB")
    
    gc.collect()
    print(f"Memory after GC: {get_memory_usage():.2f} MB")

if __name__ == "__main__":
    test_memory()
