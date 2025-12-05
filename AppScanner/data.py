"""
AppScanner Data Processing

Paper: AppScanner: Automatic Fingerprinting of Smartphone Apps from Encrypted Network Traffic
Conference: Euro S&P 2015

This module handles:
1. PCAP file parsing and flow extraction
2. Statistical feature extraction (54 features)
3. Dataset creation and loading
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import dpkt
    DPKT_AVAILABLE = True
except ImportError:
    DPKT_AVAILABLE = False

from scipy import stats as scipy_stats


@dataclass
class Flow:
    """Represents a network flow."""
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # TCP=6, UDP=17
    packets: List[Dict[str, Any]]  # List of packet info


@dataclass
class FlowFeatures:
    """Statistical features extracted from a flow."""
    features: np.ndarray  # 54-dimensional feature vector
    label: int  # Class label
    flow_id: str  # Unique identifier


class StatisticalFeatureExtractor:
    """
    Extract statistical features from packet lengths.

    Features per direction (incoming/outgoing/bidirectional):
    1. Packet count
    2. Minimum packet length
    3. Maximum packet length
    4. Mean packet length
    5. Standard deviation
    6. Variance
    7. Skewness
    8. Kurtosis
    9. Median Absolute Deviation (MAD)
    10-18. Percentiles (10, 20, 30, 40, 50, 60, 70, 80, 90)

    Total: 18 features * 3 directions = 54 features
    """

    def __init__(
        self,
        percentiles: List[int] = [10, 20, 30, 40, 50, 60, 70, 80, 90],
        min_packets: int = 7,
        max_packets: int = 260,
    ):
        self.percentiles = percentiles
        self.min_packets = min_packets
        self.max_packets = max_packets

    def extract_direction_features(self, lengths: np.ndarray) -> np.ndarray:
        """
        Extract 18 statistical features from packet lengths.

        Args:
            lengths: Array of packet lengths

        Returns:
            18-dimensional feature vector
        """
        features = []

        if len(lengths) == 0:
            # Return zeros for empty direction
            return np.zeros(18)

        # 1. Packet count
        features.append(len(lengths))

        # 2-6. Basic statistics
        features.append(np.min(lengths))
        features.append(np.max(lengths))
        features.append(np.mean(lengths))
        features.append(np.std(lengths))
        features.append(np.var(lengths))

        # 7-8. Shape statistics
        if len(lengths) >= 3:
            features.append(scipy_stats.skew(lengths))
            features.append(scipy_stats.kurtosis(lengths))
        else:
            features.extend([0.0, 0.0])

        # 9. Median Absolute Deviation
        median = np.median(lengths)
        mad = np.median(np.abs(lengths - median))
        features.append(mad)

        # 10-18. Percentiles
        for p in self.percentiles:
            features.append(np.percentile(lengths, p))

        return np.array(features)

    def extract_flow_features(
        self,
        packet_lengths: List[int],
        packet_directions: List[int],  # 1 for incoming, -1 for outgoing
    ) -> np.ndarray:
        """
        Extract 54 features from a flow.

        Args:
            packet_lengths: List of packet lengths
            packet_directions: List of directions (1=incoming, -1=outgoing)

        Returns:
            54-dimensional feature vector
        """
        lengths = np.array(packet_lengths)
        directions = np.array(packet_directions)

        # Truncate to max_packets
        if len(lengths) > self.max_packets:
            lengths = lengths[:self.max_packets]
            directions = directions[:self.max_packets]

        # Check minimum length
        if len(lengths) < self.min_packets:
            return None  # Skip short flows

        # Separate by direction
        incoming_mask = directions > 0
        outgoing_mask = directions < 0

        incoming_lengths = lengths[incoming_mask]
        outgoing_lengths = lengths[outgoing_mask]
        bidirectional_lengths = lengths  # All packets

        # Extract features for each direction
        incoming_features = self.extract_direction_features(incoming_lengths)
        outgoing_features = self.extract_direction_features(outgoing_lengths)
        bidirectional_features = self.extract_direction_features(bidirectional_lengths)

        # Concatenate: 18 * 3 = 54 features
        features = np.concatenate([
            incoming_features,
            outgoing_features,
            bidirectional_features,
        ])

        return features


class PCAPProcessor:
    """Process PCAP files to extract flows and features."""

    def __init__(
        self,
        burst_threshold: float = 1.0,
        min_flow_length: int = 7,
        max_flow_length: int = 260,
    ):
        """
        Initialize PCAP processor.

        Args:
            burst_threshold: Time threshold for burst detection (seconds)
            min_flow_length: Minimum packets per flow
            max_flow_length: Maximum packets per flow
        """
        if not DPKT_AVAILABLE:
            raise ImportError("dpkt is required for PCAP processing")

        self.burst_threshold = burst_threshold
        self.min_flow_length = min_flow_length
        self.max_flow_length = max_flow_length
        self.feature_extractor = StatisticalFeatureExtractor(
            min_packets=min_flow_length,
            max_packets=max_flow_length,
        )

    def extract_flows(self, pcap_path: str) -> List[Flow]:
        """
        Extract flows from PCAP file using dpkt.

        Args:
            pcap_path: Path to PCAP file

        Returns:
            List of Flow objects
        """
        import socket

        # Group packets by flow
        flow_packets = defaultdict(list)

        try:
            with open(pcap_path, 'rb') as f:
                # Try pcap format first, then pcapng
                try:
                    pcap = dpkt.pcap.Reader(f)
                except ValueError:
                    f.seek(0)
                    pcap = dpkt.pcapng.Reader(f)

                for ts, buf in pcap:
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                    except Exception:
                        continue

                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue

                    ip = eth.data
                    src_ip = socket.inet_ntoa(ip.src)
                    dst_ip = socket.inet_ntoa(ip.dst)

                    if isinstance(ip.data, dpkt.tcp.TCP):
                        tcp = ip.data
                        src_port, dst_port, protocol = tcp.sport, tcp.dport, 6
                    elif isinstance(ip.data, dpkt.udp.UDP):
                        udp = ip.data
                        src_port, dst_port, protocol = udp.sport, udp.dport, 17
                    else:
                        continue

                    # Normalize flow key (smaller IP first)
                    if (src_ip, src_port) < (dst_ip, dst_port):
                        flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                    else:
                        flow_key = (dst_ip, src_ip, dst_port, src_port, protocol)

                    pkt_info = {
                        'timestamp': float(ts),
                        'length': len(buf),
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                    }
                    flow_packets[flow_key].append(pkt_info)

        except Exception as e:
            print(f"Error reading {pcap_path}: {e}")
            return []

        # Convert to Flow objects
        flows = []
        for key, pkts in flow_packets.items():
            if len(pkts) < self.min_flow_length:
                continue

            # Sort by timestamp
            pkts.sort(key=lambda x: x['timestamp'])

            flow = Flow(
                src_ip=key[0],
                dst_ip=key[1],
                src_port=key[2],
                dst_port=key[3],
                protocol=key[4],
                packets=pkts,
            )
            flows.append(flow)

        return flows

    def flow_to_features(
        self,
        flow: Flow,
        client_ip: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """
        Convert flow to feature vector.

        Args:
            flow: Flow object
            client_ip: Client IP for direction detection (optional)

        Returns:
            54-dimensional feature vector or None if invalid
        """
        if client_ip is None:
            # Assume first packet's source is client
            client_ip = flow.packets[0]['src_ip']

        # Extract lengths and directions
        lengths = []
        directions = []

        for pkt in flow.packets:
            lengths.append(pkt['length'])
            # Incoming: from server to client (dst_ip = client)
            # Outgoing: from client to server (src_ip = client)
            if pkt['src_ip'] == client_ip:
                directions.append(-1)  # Outgoing
            else:
                directions.append(1)   # Incoming

        return self.feature_extractor.extract_flow_features(lengths, directions)

    def process_pcap(
        self,
        pcap_path: str,
        label: int,
        client_ip: Optional[str] = None,
    ) -> List[FlowFeatures]:
        """
        Process PCAP file and extract features.

        Args:
            pcap_path: Path to PCAP file
            label: Class label for this file
            client_ip: Client IP address

        Returns:
            List of FlowFeatures objects
        """
        flows = self.extract_flows(pcap_path)
        results = []

        for i, flow in enumerate(flows):
            features = self.flow_to_features(flow, client_ip)
            if features is not None:
                flow_id = f"{os.path.basename(pcap_path)}_{i}"
                results.append(FlowFeatures(
                    features=features,
                    label=label,
                    flow_id=flow_id,
                ))

        return results


class AppScannerDataset(Dataset):
    """PyTorch Dataset for AppScanner."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        normalize: bool = True,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ):
        """
        Initialize dataset.

        Args:
            features: Feature matrix (N x 54)
            labels: Label array (N,)
            normalize: Whether to normalize features
            mean: Pre-computed mean for normalization
            std: Pre-computed std for normalization
        """
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)

        if normalize:
            if mean is None:
                self.mean = self.features.mean(axis=0)
            else:
                self.mean = mean

            if std is None:
                self.std = self.features.std(axis=0)
                self.std[self.std == 0] = 1.0  # Avoid division by zero
            else:
                self.std = std

            self.features = (self.features - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx]),
        )

    def get_normalization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and std for normalization."""
        return self.mean, self.std


def create_dataset_from_directory(
    data_dir: str,
    min_flow_length: int = 7,
    max_flow_length: int = 260,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Create dataset from directory structure.

    Expected structure:
    data_dir/
    ├── class1/
    │   ├── file1.pcap
    │   └── file2.pcap
    ├── class2/
    │   └── ...
    └── ...

    Args:
        data_dir: Root directory
        min_flow_length: Minimum packets per flow
        max_flow_length: Maximum packets per flow

    Returns:
        features: Feature matrix
        labels: Label array
        label_map: Mapping from label to class name
    """
    processor = PCAPProcessor(
        min_flow_length=min_flow_length,
        max_flow_length=max_flow_length,
    )

    all_features = []
    all_labels = []
    label_map = {}
    label_idx = 0

    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        label_map[label_idx] = class_name
        print(f"Processing class {class_name} (label={label_idx})")

        for filename in os.listdir(class_dir):
            if not filename.endswith('.pcap'):
                continue

            pcap_path = os.path.join(class_dir, filename)
            try:
                flow_features = processor.process_pcap(pcap_path, label_idx)
                for ff in flow_features:
                    all_features.append(ff.features)
                    all_labels.append(ff.label)
            except Exception as e:
                print(f"Error processing {pcap_path}: {e}")

        label_idx += 1

    features = np.array(all_features)
    labels = np.array(all_labels)

    print(f"Total flows: {len(labels)}")
    print(f"Number of classes: {len(label_map)}")

    return features, labels, label_map


def create_dataset_from_csv(
    csv_path: str,
    feature_cols: Optional[List[str]] = None,
    label_col: str = 'label',
) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Create dataset from CSV file with pre-extracted features.

    Args:
        csv_path: Path to CSV file
        feature_cols: Column names for features (None = all except label)
        label_col: Column name for labels

    Returns:
        features: Feature matrix
        labels: Label array
        label_map: Mapping from label to class name
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Get feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]

    features = df[feature_cols].values.astype(np.float32)

    # Encode labels
    unique_labels = df[label_col].unique()
    label_map = {i: str(label) for i, label in enumerate(sorted(unique_labels))}
    label_to_idx = {v: k for k, v in label_map.items()}
    labels = df[label_col].map(label_to_idx).values.astype(np.int64)

    return features, labels, label_map


def create_dataloaders(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int = 128,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, np.ndarray]]:
    """
    Create train, validation, and test dataloaders.

    Default split ratio is 8:1:1 (train:val:test) by flow.

    Args:
        features: Feature matrix
        labels: Label array
        batch_size: Batch size
        train_ratio: Training set ratio (default 0.8)
        val_ratio: Validation set ratio (default 0.1)
        test_ratio: Test set ratio (default 0.1)
        seed: Random seed
        num_workers: Number of data loading workers

    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader
        norm_params: (mean, std) for normalization
    """
    np.random.seed(seed)

    # Shuffle indices
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)

    # Split with 8:1:1 ratio
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    # n_test = n_samples - n_train - n_val (remaining)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    # Create datasets
    train_dataset = AppScannerDataset(
        features[train_indices],
        labels[train_indices],
        normalize=True,
    )

    # Use training set's normalization params for val/test
    mean, std = train_dataset.get_normalization_params()

    val_dataset = AppScannerDataset(
        features[val_indices],
        labels[val_indices],
        normalize=True,
        mean=mean,
        std=std,
    )

    test_dataset = AppScannerDataset(
        features[test_indices],
        labels[test_indices],
        normalize=True,
        mean=mean,
        std=std,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, (mean, std)


def save_dataset(
    features: np.ndarray,
    labels: np.ndarray,
    label_map: Dict[int, str],
    save_path: str,
):
    """Save processed dataset to file."""
    data = {
        'features': features,
        'labels': labels,
        'label_map': label_map,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Dataset saved to {save_path}")


def load_dataset(load_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    """Load processed dataset from file (combined all splits).

    Supports three formats:
    1. Old format: {'features', 'labels', 'label_map'}
    2. Train/test format: {'train_features', 'train_labels', 'test_features', 'test_labels', 'label_map'}
    3. Train/val/test format: {'train_features', 'val_features', 'test_features', ...}
    """
    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    # Check format and load accordingly
    if 'features' in data:
        # Old format
        features = data['features']
        labels = data['labels']
        label_map = data['label_map']
    elif 'train_features' in data:
        # New format from iscxvpn_processor.py - combine all splits
        parts_features = [data['train_features']]
        parts_labels = [data['train_labels']]

        if 'val_features' in data:
            parts_features.append(data['val_features'])
            parts_labels.append(data['val_labels'])

        parts_features.append(data['test_features'])
        parts_labels.append(data['test_labels'])

        features = np.concatenate(parts_features, axis=0)
        labels = np.concatenate(parts_labels, axis=0)
        label_map = data['label_map']
    else:
        raise KeyError(f"Unknown dataset format. Keys: {list(data.keys())}")

    # Clean data: replace NaN/Inf with 0
    nan_count = np.isnan(features).sum()
    inf_count = np.isinf(features).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"Warning: Found {nan_count} NaN and {inf_count} Inf values, replacing with 0")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features, labels, label_map


def load_dataset_split(load_path: str) -> Tuple[
    Tuple[np.ndarray, np.ndarray],  # train
    Tuple[np.ndarray, np.ndarray],  # val
    Tuple[np.ndarray, np.ndarray],  # test
    Dict[int, str],  # label_map
]:
    """Load processed dataset with pre-defined splits (train/val/test).

    Returns:
        train_data: (train_features, train_labels)
        val_data: (val_features, val_labels)
        test_data: (test_features, test_labels)
        label_map: class label mapping
    """
    with open(load_path, 'rb') as f:
        data = pickle.load(f)

    if 'train_features' not in data:
        raise KeyError("Dataset does not contain pre-split data. Use load_dataset() instead.")

    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    test_labels = data['test_labels']
    label_map = data['label_map']

    # Check for validation set
    if 'val_features' in data:
        val_features = data['val_features']
        val_labels = data['val_labels']
    else:
        # No validation set, create empty arrays
        val_features = np.array([]).reshape(0, train_features.shape[1])
        val_labels = np.array([], dtype=train_labels.dtype)

    # Clean data: replace NaN/Inf with 0
    for arr_name, arr in [('train', train_features), ('val', val_features), ('test', test_features)]:
        nan_count = np.isnan(arr).sum()
        inf_count = np.isinf(arr).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"Warning: Found {nan_count} NaN and {inf_count} Inf in {arr_name}, replacing with 0")

    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    val_features = np.nan_to_num(val_features, nan=0.0, posinf=0.0, neginf=0.0)
    test_features = np.nan_to_num(test_features, nan=0.0, posinf=0.0, neginf=0.0)

    return (train_features, train_labels), (val_features, val_labels), (test_features, test_labels), label_map


def create_dataloaders_from_split(
    train_data: Tuple[np.ndarray, np.ndarray],
    val_data: Tuple[np.ndarray, np.ndarray],
    test_data: Tuple[np.ndarray, np.ndarray],
    batch_size: int = 128,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, np.ndarray]]:
    """
    Create dataloaders from pre-split data.

    Args:
        train_data: (features, labels) for training
        val_data: (features, labels) for validation
        test_data: (features, labels) for testing
        batch_size: Batch size
        num_workers: Number of data loading workers

    Returns:
        train_loader, val_loader, test_loader, (mean, std)
    """
    train_features, train_labels = train_data
    val_features, val_labels = val_data
    test_features, test_labels = test_data

    # Create training dataset and compute normalization params
    train_dataset = AppScannerDataset(
        train_features,
        train_labels,
        normalize=True,
    )
    mean, std = train_dataset.get_normalization_params()

    # Create val/test datasets with same normalization
    val_dataset = AppScannerDataset(
        val_features,
        val_labels,
        normalize=True,
        mean=mean,
        std=std,
    )

    test_dataset = AppScannerDataset(
        test_features,
        test_labels,
        normalize=True,
        mean=mean,
        std=std,
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, (mean, std)


if __name__ == '__main__':
    # Test feature extraction
    extractor = StatisticalFeatureExtractor()

    # Simulate a flow with random packet lengths
    np.random.seed(42)
    packet_lengths = np.random.randint(64, 1500, 50).tolist()
    packet_directions = np.random.choice([1, -1], 50).tolist()

    features = extractor.extract_flow_features(packet_lengths, packet_directions)
    print(f"Feature vector shape: {features.shape}")
    print(f"Features: {features}")

    # Test dataset
    n_samples = 1000
    n_features = 54
    n_classes = 10

    features = np.random.randn(n_samples, n_features).astype(np.float32)
    labels = np.random.randint(0, n_classes, n_samples)

    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        features, labels, batch_size=64
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    for x, y in train_loader:
        print(f"Batch shape: {x.shape}, {y.shape}")
        break

    print("Data processing tests passed!")
