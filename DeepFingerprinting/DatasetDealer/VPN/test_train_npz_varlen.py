#!/usr/bin/env python3
"""
Comprehensive test suite for train_npz_varlen.py
Tests edge cases and verifies all components work correctly.
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from train_npz_varlen import (
    LazyNPZDataset, LRUCache, collate_fn, 
    load_label_mapping, collect_npz_files,
    split_dataset, create_model
)

def test_lru_cache():
    """Test LRU cache functionality."""
    print("Testing LRU cache...")
    cache = LRUCache(capacity=3)
    
    # Add items
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") == 3
    
    # Add one more, should evict "a" (least recently used)
    cache.put("d", 4)
    assert cache.get("a") is None
    assert cache.get("d") == 4
    
    print("✓ LRU cache test passed")

def test_load_label_mapping():
    """Test label mapping loading."""
    print("Testing label mapping...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "label2id": {"site1": 0, "site2": 1, "site3": 2},
            "id2label": {"0": "site1", "1": "site2", "2": "site3"}
        }, f)
        temp_path = Path(f.name)
    
    try:
        label2id, id2label, num_classes = load_label_mapping(temp_path)
        assert num_classes == 3
        assert label2id["site1"] == 0
        assert id2label["1"] == "site2"
        print("✓ Label mapping test passed")
    finally:
        temp_path.unlink()

def test_collate_fn():
    """Test collate function."""
    print("Testing collate function...")
    
    # Create batch of samples
    batch = [
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), 0),
        (np.array([4.0, 5.0, 6.0], dtype=np.float32), 1),
        (np.array([7.0, 8.0, 9.0], dtype=np.float32), 2),
    ]
    
    flows, labels = collate_fn(batch)
    
    assert flows.shape == (3, 3)
    assert labels.shape == (3,)
    assert torch.allclose(flows[0], torch.tensor([1.0, 2.0, 3.0]))
    assert labels[1] == 1
    
    print("✓ Collate function test passed")

def test_lazy_npz_dataset():
    """Test LazyNPZDataset with variable-length flows."""
    print("Testing LazyNPZDataset...")
    
    # Create temporary directory with test NPZ files
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Create test NPZ file
        test_npz = temp_dir / "test.npz"
        
        # Create variable-length flows
        flows = np.empty(5, dtype=object)
        labels = np.empty(5, dtype=object)
        
        flows[0] = np.array([-1, 1], dtype=np.int8)
        flows[1] = np.array([-1, 1, -1, 1], dtype=np.int8)
        flows[2] = np.array([1] * 100, dtype=np.int8)
        flows[3] = np.array([-1] * 10000, dtype=np.int8)  # Very long
        flows[4] = np.array([1, -1], dtype=np.int8)
        
        for i in range(5):
            labels[i] = "test_site"
        
        np.savez(test_npz, flows=flows, labels=labels)
        
        # Create dataset
        label2id = {"test_site": 0}
        dataset = LazyNPZDataset(
            [test_npz],
            label2id,
            max_len=100,
            cache_size=1,
            shuffle=False
        )
        
        # Iterate through dataset
        samples = list(dataset)
        assert len(samples) == 5
        
        # Check first sample (short, should be padded)
        flow0, label0 = samples[0]
        assert len(flow0) == 100
        assert label0 == 0
        assert flow0[0] == -1.0 and flow0[1] == 1.0
        assert all(flow0[2:] == 0.0)  # Padded with zeros
        
        # Check sample with length > max_len (should be truncated)
        flow3, label3 = samples[3]
        assert len(flow3) == 100  # Truncated
        assert all(flow3 == -1.0)
        
        print("✓ LazyNPZDataset test passed")
        
    finally:
        shutil.rmtree(temp_dir)

def test_split_dataset():
    """Test dataset splitting."""
    print("Testing dataset splitting...")
    
    # Create fake paths
    paths = [Path(f"file_{i}.npz") for i in range(100)]
    
    train, val, test = split_dataset(paths, train_ratio=0.8, val_ratio=0.1, seed=42)
    
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10
    assert len(set(train) & set(val)) == 0  # No overlap
    assert len(set(train) & set(test)) == 0  # No overlap
    
    print("✓ Dataset splitting test passed")

def test_create_model():
    """Test model creation."""
    print("Testing model creation...")
    
    model = create_model(num_classes=10, seq_len=5000)
    
    # Check output layer
    assert model.classifier.out_features == 10
    
    # Test forward pass
    x = torch.randn(2, 5000)  # (batch_size, seq_len)
    output = model(x)
    assert output.shape == (2, 10)
    
    print("✓ Model creation test passed")

def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test with empty flow
        test_npz = temp_dir / "empty.npz"
        flows = np.empty(1, dtype=object)
        labels = np.empty(1, dtype=object)
        flows[0] = np.array([], dtype=np.int8)  # Empty flow
        labels[0] = "site1"
        np.savez(test_npz, flows=flows, labels=labels)
        
        label2id = {"site1": 0}
        dataset = LazyNPZDataset([test_npz], label2id, max_len=10, shuffle=False)
        samples = list(dataset)
        assert len(samples) == 1
        assert len(samples[0][0]) == 10  # Should be padded to max_len
        
        # Test with single element flow
        test_npz2 = temp_dir / "single.npz"
        flows = np.empty(1, dtype=object)
        labels = np.empty(1, dtype=object)
        flows[0] = np.array([1], dtype=np.int8)
        labels[0] = "site1"
        np.savez(test_npz2, flows=flows, labels=labels)
        
        dataset2 = LazyNPZDataset([test_npz2], label2id, max_len=10, shuffle=False)
        samples2 = list(dataset2)
        assert samples2[0][0][0] == 1.0
        
        print("✓ Edge cases test passed")
        
    finally:
        shutil.rmtree(temp_dir)

def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running comprehensive test suite for train_npz_varlen.py")
    print("="*60)
    print()
    
    try:
        test_lru_cache()
        test_load_label_mapping()
        test_collate_fn()
        test_lazy_npz_dataset()
        test_split_dataset()
        test_create_model()
        test_edge_cases()
        
        print()
        print("="*60)
        print("✓ All tests passed successfully!")
        print("="*60)
        return True
        
    except Exception as e:
        print()
        print("="*60)
        print(f"✗ Test failed: {e}")
        print("="*60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
