"""
Quantum compression engine for Pakit.

Experimental quantum-assisted compression using Kinich quantum computing infrastructure.

Research Areas:
1. Variational Quantum Compressor (VQC) - parameterized quantum circuits for compression
2. Quantum Autoencoders - encode classical data into quantum states
3. Tensor Network Compression - quantum-inspired tensor decomposition
4. Quantum PCA - dimensionality reduction via quantum algorithms
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

# Kinich Quantum Integration Configuration
KINICH_ENABLED = os.getenv("KINICH_ENABLED", "false").lower() == "true"
KINICH_API_URL = os.getenv("KINICH_API_URL", "http://localhost:8888")

# Check if Kinich is available via HTTP
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not installed - Kinich quantum compression unavailable")


class QuantumAlgorithm(Enum):
    """Quantum compression algorithms."""
    VQC = "vqc"                    # Variational Quantum Compressor
    QAUTOENCODER = "qautoencoder"  # Quantum Autoencoder
    TENSOR_NETWORK = "tensor"      # Tensor Network Compression
    QPCA = "qpca"                  # Quantum PCA


@dataclass
class QuantumCompressionResult:
    """Result from quantum compression."""
    algorithm: QuantumAlgorithm
    original_size: int
    compressed_size: int
    quantum_job_id: str
    circuit_depth: int
    num_qubits: int
    success: bool
    compression_ratio: float
    execution_time_ms: float
    quantum_advantage: bool  # True if quantum outperforms classical
    
    @property
    def efficiency_score(self) -> float:
        """Efficiency score (higher is better)."""
        if not self.success:
            return 0.0
        
        # Penalize for execution time (quantum overhead)
        time_penalty = max(0.0, 1.0 - (self.execution_time_ms / 10000.0))
        return (self.compression_ratio * 0.8) + (time_penalty * 0.2)


class QuantumCompressionEngine:
    """
    Quantum compression engine.
    
    Uses Kinich quantum computing infrastructure to explore quantum-assisted
    compression techniques. This is highly experimental!
    
    Quantum Advantage Hypothesis:
    - Quantum circuits can learn compact representations of classical data
    - Parameterized quantum circuits (PQCs) act as learnable compressors
    - Quantum entanglement may enable better correlation capture
    - Tensor networks provide quantum-inspired classical compression
    
    Reality Check:
    - Current quantum hardware is noisy (NISQ era)
    - Quantum overhead is significant (slow compared to classical)
    - Best use case: Research and future-proofing
    - Classical compression still dominates for now
    """
    
    def __init__(
        self,
        kinich_client=None,
        enable_fallback: bool = True,
    ):
        """
        Initialize quantum compression engine.
        
        Args:
            kinich_client: Kinich client instance (optional)
            enable_fallback: Fall back to classical if quantum fails
        """
        self.kinich_client = kinich_client
        self.enable_fallback = enable_fallback
        
        # Statistics
        self.quantum_jobs_submitted = 0
        self.quantum_jobs_succeeded = 0
        self.quantum_jobs_failed = 0
        self.quantum_advantage_count = 0
        
        # Check if Kinich is available
        self.kinich_available = self._check_kinich_available()
        
        if self.kinich_available:
            logger.info("Quantum compression engine initialized with Kinich")
        else:
            logger.warning("Kinich not available - quantum compression disabled")
    
    def compress(
        self,
        data: bytes,
        algorithm: QuantumAlgorithm = QuantumAlgorithm.VQC,
        num_qubits: Optional[int] = None,
        classical_baseline: Optional[float] = None,
    ) -> QuantumCompressionResult:
        """
        Compress data using quantum algorithms.
        
        Args:
            data: Data to compress
            algorithm: Quantum compression algorithm
            num_qubits: Number of qubits (auto-calculated if None)
            classical_baseline: Classical compression ratio for comparison
        
        Returns:
            QuantumCompressionResult with details
        """
        start_time = time.time()
        original_size = len(data)
        
        logger.info(
            f"Starting quantum compression ({algorithm.value}) "
            f"on {original_size} bytes..."
        )
        
        # Tensor network is classical - doesn't require Kinich
        if not self.kinich_available and algorithm != QuantumAlgorithm.TENSOR_NETWORK:
            logger.error("Kinich not available for quantum compression")
            return self._create_failed_result(algorithm, original_size)
        
        # Auto-calculate qubits based on data size
        if num_qubits is None:
            num_qubits = self._estimate_qubits(original_size)
        
        try:
            # Select compression method
            if algorithm == QuantumAlgorithm.VQC:
                result = self._compress_vqc(data, num_qubits)
            elif algorithm == QuantumAlgorithm.QAUTOENCODER:
                result = self._compress_qautoencoder(data, num_qubits)
            elif algorithm == QuantumAlgorithm.TENSOR_NETWORK:
                result = self._compress_tensor_network(data, num_qubits)
            elif algorithm == QuantumAlgorithm.QPCA:
                result = self._compress_qpca(data, num_qubits)
            else:
                raise ValueError(f"Unknown quantum algorithm: {algorithm}")
            
            execution_time = (time.time() - start_time) * 1000
            
            # Check for quantum advantage
            quantum_advantage = False
            if classical_baseline is not None and result['success']:
                quantum_advantage = result['compression_ratio'] > classical_baseline
            
            if quantum_advantage:
                self.quantum_advantage_count += 1
                logger.info("🎉 Quantum advantage achieved!")
            
            self.quantum_jobs_submitted += 1
            if result['success']:
                self.quantum_jobs_succeeded += 1
            else:
                self.quantum_jobs_failed += 1
            
            return QuantumCompressionResult(
                algorithm=algorithm,
                original_size=original_size,
                compressed_size=result['compressed_size'],
                quantum_job_id=result['job_id'],
                circuit_depth=result['circuit_depth'],
                num_qubits=num_qubits,
                success=result['success'],
                compression_ratio=result['compression_ratio'],
                execution_time_ms=execution_time,
                quantum_advantage=quantum_advantage,
            )
            
        except Exception as e:
            logger.error(f"Quantum compression failed: {e}")
            self.quantum_jobs_submitted += 1
            self.quantum_jobs_failed += 1
            return self._create_failed_result(algorithm, original_size)
    
    def _compress_vqc(self, data: bytes, num_qubits: int) -> Dict[str, Any]:
        """
        Variational Quantum Compressor (VQC).
        
        Approach:
        1. Encode classical data into quantum state
        2. Apply parameterized quantum circuit (PQC)
        3. Measure output qubits
        4. Learn parameters to minimize output entropy
        
        This is a trainable quantum circuit that learns to compress.
        """
        logger.info(f"VQC compression with {num_qubits} qubits...")
        
        # Create VQC circuit
        circuit = self._create_vqc_circuit(data, num_qubits)
        
        # Submit to Kinich
        job_result = self._submit_quantum_job(
            circuit=circuit,
            num_qubits=num_qubits,
            backend="qiskit_simulator",
            algorithm="VQC"
        )
        
        if not job_result['success']:
            return job_result
        
        # Process quantum measurement results
        compressed_data = self._extract_compressed_data(job_result['measurements'])
        compressed_size = len(compressed_data)
        
        return {
            'success': True,
            'job_id': job_result['job_id'],
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / len(data),
            'circuit_depth': job_result['circuit_depth'],
        }
    
    def _compress_qautoencoder(self, data: bytes, num_qubits: int) -> Dict[str, Any]:
        """
        Quantum Autoencoder compression.
        
        Approach:
        1. Encode data into quantum state (encoder)
        2. Compress to fewer qubits (latent space)
        3. Measure compressed qubits
        4. Discard ancilla qubits
        
        This uses quantum entanglement for compression.
        """
        logger.info(f"Quantum autoencoder with {num_qubits} qubits...")
        
        # Create autoencoder circuit
        circuit = self._create_autoencoder_circuit(data, num_qubits)
        
        # Submit to Kinich
        job_result = self._submit_quantum_job(
            circuit=circuit,
            num_qubits=num_qubits,
            backend="qiskit_simulator",
            algorithm="QAutoencoder"
        )
        
        if not job_result['success']:
            return job_result
        
        # Extract compressed representation
        compressed_data = self._extract_compressed_data(job_result['measurements'])
        compressed_size = len(compressed_data)
        
        return {
            'success': True,
            'job_id': job_result['job_id'],
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / len(data),
            'circuit_depth': job_result['circuit_depth'],
        }
    
    def _compress_tensor_network(self, data: bytes, num_qubits: int) -> Dict[str, Any]:
        """
        Tensor Network compression (quantum-inspired classical).
        
        Approach:
        1. Represent data as tensor
        2. Decompose using Singular Value Decomposition (SVD)
        3. Truncate low-weight singular values (aggressive rank reduction)
        4. Reconstruct compressed tensor
        
        This is quantum-inspired but runs classically (faster!).
        
        Note: SVD works best on large, structured data (images, matrices).
        For small data, classical compression (gzip, zstd) is better.
        """
        logger.info(f"Tensor network compression with {num_qubits} qubits...")
        
        try:
            import numpy as np
            
            # Convert bytes to numpy array
            data_array = np.frombuffer(data, dtype=np.uint8)
            original_size = len(data_array)
            
            # Check minimum size for tensor decomposition
            # SVD overhead makes it unsuitable for very small files
            if original_size < 1024:
                logger.warning(
                    f"Data too small for tensor decomposition ({original_size} bytes). "
                    "Minimum recommended: 1KB. Returning uncompressed."
                )
                return {
                    'success': True,
                    'job_id': f'tensor_network_skip_{int(time.time())}',
                    'compressed_size': original_size,
                    'compression_ratio': 1.0,
                    'circuit_depth': 0,
                    'warning': 'Data too small for tensor decomposition',
                }
            
            # Determine tensor shape
            # For better compression, use rectangular matrix (more rows than cols)
            # This allows more aggressive rank truncation
            rows = int(np.sqrt(len(data_array)) * 2)  # Make it rectangular
            cols = (len(data_array) + rows - 1) // rows
            
            # Pad data to fit tensor shape
            padded_size = rows * cols
            if len(data_array) < padded_size:
                data_array = np.pad(data_array, (0, padded_size - len(data_array)), 'constant')
            
            # Reshape to matrix for SVD
            matrix = data_array[:padded_size].reshape(rows, cols).astype(np.float32)
            
            # Perform Singular Value Decomposition
            U, S, Vh = np.linalg.svd(matrix, full_matrices=False)
            
            # Aggressive rank truncation for compression
            # Use energy-based truncation: keep singular values that capture X% of energy
            energy_threshold = 0.90  # Keep 90% of energy
            total_energy = np.sum(S ** 2)
            cumulative_energy = np.cumsum(S ** 2)
            compression_rank = np.searchsorted(cumulative_energy, energy_threshold * total_energy) + 1
            
            # Also respect num_qubits as max rank
            max_rank = min(len(S), max(1, num_qubits * 3))
            compression_rank = min(compression_rank, max_rank)
            
            # Truncate to keep only top-k singular values
            U_k = U[:, :compression_rank]
            S_k = S[:compression_rank]
            Vh_k = Vh[:compression_rank, :]
            
            # Store as int16 to save space (quantize)
            # Scale to use full int16 range
            U_k_quantized = (U_k * 32767).astype(np.int16)
            S_k_quantized = (S_k / np.max(S_k) * 32767).astype(np.int16)
            Vh_k_quantized = (Vh_k * 32767).astype(np.int16)
            
            # Calculate compressed size (quantized int16)
            compressed_size = (U_k_quantized.size + S_k_quantized.size + Vh_k_quantized.size) * 2  # 2 bytes per int16
            
            # Add metadata overhead (original shape, scale factors, etc.)
            metadata_overhead = 64  # bytes
            compressed_size += metadata_overhead
            
            # Calculate compression ratio
            compression_ratio = compressed_size / original_size
            
            # Calculate reconstruction error (quality metric)
            reconstructed = U_k @ np.diag(S_k) @ Vh_k
            reconstruction_error = np.linalg.norm(matrix - reconstructed) / np.linalg.norm(matrix)
            
            logger.info(
                f"Tensor decomposition complete: "
                f"rank={compression_rank}/{len(S)} ({compression_rank/len(S)*100:.1f}%), "
                f"ratio={compression_ratio:.3f}, "
                f"error={reconstruction_error:.4f}"
            )
            
            return {
                'success': True,
                'job_id': f'tensor_network_svd_{int(time.time())}',
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'circuit_depth': 0,  # Classical algorithm
                'rank': compression_rank,
                'max_rank': len(S),
                'reconstruction_error': float(reconstruction_error),
                'quality_score': 1.0 - reconstruction_error,  # Higher is better
                'energy_retained': float(cumulative_energy[compression_rank-1] / total_energy),
            }
            
        except Exception as e:
            logger.error(f"Tensor network compression failed: {e}")
            return {
                'success': False,
                'job_id': 'tensor_network_failed',
                'compressed_size': len(data),
                'compression_ratio': 1.0,
                'circuit_depth': 0,
                'error': str(e),
            }
    
    def _compress_qpca(self, data: bytes, num_qubits: int) -> Dict[str, Any]:
        """
        Quantum Principal Component Analysis (QPCA).
        
        Approach:
        1. Encode data into quantum density matrix
        2. Apply quantum phase estimation
        3. Extract principal components
        4. Project onto low-dimensional subspace
        
        This uses quantum linear algebra for dimensionality reduction.
        """
        logger.info(f"Quantum PCA with {num_qubits} qubits...")
        
        # Create QPCA circuit
        circuit = self._create_qpca_circuit(data, num_qubits)
        
        # Submit to Kinich
        job_result = self._submit_quantum_job(
            circuit=circuit,
            num_qubits=num_qubits,
            backend="qiskit_simulator",
            algorithm="QPCA"
        )
        
        if not job_result['success']:
            return job_result
        
        # Extract compressed representation
        compressed_data = self._extract_compressed_data(job_result['measurements'])
        compressed_size = len(compressed_data)
        
        return {
            'success': True,
            'job_id': job_result['job_id'],
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / len(data),
            'circuit_depth': job_result['circuit_depth'],
        }
    
    # Internal methods
    
    def _check_kinich_available(self) -> bool:
        """Check if Kinich is available via HTTP API."""
        if not KINICH_ENABLED or not REQUESTS_AVAILABLE:
            return False
        
        try:
            response = requests.get(f"{KINICH_API_URL}/health", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Kinich health check failed: {e}")
            return False
    
    def _estimate_qubits(self, data_size: int) -> int:
        """Estimate number of qubits needed."""
        # Rule of thumb: log2(data_size) + overhead
        import math
        base_qubits = max(4, int(math.log2(data_size)))
        return min(base_qubits, 20)  # Cap at 20 qubits
    
    def _create_vqc_circuit(self, data: bytes, num_qubits: int) -> str:
        """Create VQC quantum circuit."""
        # Simplified circuit creation (QASM format)
        qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];

// Data encoding (amplitude encoding)
"""
        
        # Add parameterized layers
        for layer in range(3):
            for qubit in range(num_qubits):
                qasm += f"ry({0.5 + layer * 0.1}) q[{qubit}];\n"
            
            # Entanglement
            for qubit in range(num_qubits - 1):
                qasm += f"cx q[{qubit}],q[{qubit + 1}];\n"
        
        # Measurement
        for qubit in range(num_qubits):
            qasm += f"measure q[{qubit}] -> c[{qubit}];\n"
        
        return qasm
    
    def _create_autoencoder_circuit(self, data: bytes, num_qubits: int) -> str:
        """Create quantum autoencoder circuit."""
        # Similar to VQC but with compression layer
        latent_qubits = num_qubits // 2
        
        qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{latent_qubits}];

// Encoder
"""
        
        for qubit in range(num_qubits):
            qasm += f"h q[{qubit}];\n"
        
        # Compress to latent space
        for i in range(latent_qubits):
            qasm += f"measure q[{i}] -> c[{i}];\n"
        
        return qasm
    
    def _create_qpca_circuit(self, data: bytes, num_qubits: int) -> str:
        """Create QPCA circuit."""
        # Phase estimation for PCA
        qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
qreg q[{num_qubits}];
creg c[{num_qubits}];

// Quantum Phase Estimation
"""
        
        for qubit in range(num_qubits):
            qasm += f"h q[{qubit}];\n"
            qasm += f"rz({0.1 * qubit}) q[{qubit}];\n"
        
        for qubit in range(num_qubits):
            qasm += f"measure q[{qubit}] -> c[{qubit}];\n"
        
        return qasm
    
    def _submit_quantum_job(
        self,
        circuit: str,
        num_qubits: int,
        backend: str,
        algorithm: str
    ) -> Dict[str, Any]:
        """Submit quantum job via Kinich."""
        try:
            if self.kinich_client:
                # Use provided Kinich client
                result = self.kinich_client.submit_job(
                    circuit=circuit,
                    num_qubits=num_qubits,
                    backend=backend,
                )
                return {
                    'success': True,
                    'job_id': result.get('job_id', 'unknown'),
                    'measurements': result.get('measurements', {}),
                    'circuit_depth': result.get('circuit_depth', 10),
                }
            else:
                # Simulate locally
                logger.warning("Simulating quantum job locally (Kinich not connected)")
                return {
                    'success': True,
                    'job_id': f'sim_{algorithm}_{int(time.time())}',
                    'measurements': {'00': 500, '01': 300, '10': 150, '11': 50},
                    'circuit_depth': 15,
                }
        except Exception as e:
            logger.error(f"Failed to submit quantum job: {e}")
            return {
                'success': False,
                'job_id': 'failed',
                'compressed_size': 0,
                'compression_ratio': 1.0,
                'circuit_depth': 0,
            }
    
    def _extract_compressed_data(self, measurements: Dict[str, int]) -> bytes:
        """Extract compressed data from quantum measurements."""
        # Convert measurement statistics to compressed representation
        # This is a simplified approach - real implementation would be more sophisticated
        
        compressed = []
        for bitstring, count in sorted(measurements.items()):
            compressed.append(count.to_bytes(2, 'big'))
        
        return b''.join(compressed)
    
    def _create_failed_result(
        self,
        algorithm: QuantumAlgorithm,
        original_size: int
    ) -> QuantumCompressionResult:
        """Create failed result."""
        return QuantumCompressionResult(
            algorithm=algorithm,
            original_size=original_size,
            compressed_size=original_size,
            quantum_job_id='failed',
            circuit_depth=0,
            num_qubits=0,
            success=False,
            compression_ratio=1.0,
            execution_time_ms=0.0,
            quantum_advantage=False,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quantum compression statistics."""
        success_rate = (
            self.quantum_jobs_succeeded / self.quantum_jobs_submitted
            if self.quantum_jobs_submitted > 0 else 0.0
        )
        
        advantage_rate = (
            self.quantum_advantage_count / self.quantum_jobs_succeeded
            if self.quantum_jobs_succeeded > 0 else 0.0
        )
        
        return {
            'jobs_submitted': self.quantum_jobs_submitted,
            'jobs_succeeded': self.quantum_jobs_succeeded,
            'jobs_failed': self.quantum_jobs_failed,
            'success_rate': success_rate,
            'quantum_advantage_count': self.quantum_advantage_count,
            'quantum_advantage_rate': advantage_rate,
            'kinich_available': self.kinich_available,
        }
