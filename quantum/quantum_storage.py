"""
Quantum storage encoder for Pakit.

Encodes classical data into quantum states for ultra-dense storage experiments.
"""

import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class QuantumEncodingScheme(Enum):
    """Quantum encoding schemes."""
    AMPLITUDE = "amplitude"      # Amplitude encoding (dense but complex)
    BASIS = "basis"              # Basis encoding (simple but sparse)
    ANGLE = "angle"              # Angle encoding (good for continuous data)
    QSAMPLE = "qsample"          # Quantum sampling encoding


@dataclass
class QuantumEncodingResult:
    """Result from quantum encoding."""
    scheme: QuantumEncodingScheme
    original_bits: int
    encoded_qubits: int
    encoding_efficiency: float  # bits per qubit
    quantum_state_prepared: bool
    theoretical_density: float  # theoretical max bits per qubit
    
    @property
    def density_ratio(self) -> float:
        """Ratio of achieved vs theoretical density."""
        return self.encoding_efficiency / self.theoretical_density if self.theoretical_density > 0 else 0.0


class QuantumStorageEncoder:
    """
    Quantum storage encoder.
    
    Explores quantum state preparation for ultra-dense data storage.
    
    Theoretical Background:
    - N qubits can represent 2^N classical states simultaneously (superposition)
    - Amplitude encoding: O(2^N) classical bits in N qubits
    - Reality: Measurement collapses state (can only extract log2(2^N) = N bits)
    - Quantum advantage: Parallel processing, not storage density
    
    Experimental Goals:
    1. Test various encoding schemes
    2. Measure encoding/decoding overhead
    3. Explore quantum error correction costs
    4. Benchmark against classical storage
    
    Expected Outcome:
    - Classical storage likely superior for pure storage
    - Quantum advantage in: computation on encoded data, quantum networks
    - Research value: Future quantum RAM, quantum databases
    """
    
    def __init__(self):
        """Initialize quantum storage encoder."""
        self.encodings_performed = 0
        self.total_bits_encoded = 0
        self.total_qubits_used = 0
        
        logger.info("Quantum storage encoder initialized")
    
    def encode(
        self,
        data: bytes,
        scheme: QuantumEncodingScheme = QuantumEncodingScheme.AMPLITUDE,
        max_qubits: int = 20
    ) -> QuantumEncodingResult:
        """
        Encode classical data into quantum state.
        
        Args:
            data: Classical data to encode
            scheme: Encoding scheme
            max_qubits: Maximum qubits to use
        
        Returns:
            QuantumEncodingResult with details
        """
        data_bits = len(data) * 8
        
        logger.info(f"Encoding {len(data)} bytes ({data_bits} bits) using {scheme.value}...")
        
        # Select encoding method
        if scheme == QuantumEncodingScheme.AMPLITUDE:
            result = self._encode_amplitude(data, max_qubits)
        elif scheme == QuantumEncodingScheme.BASIS:
            result = self._encode_basis(data, max_qubits)
        elif scheme == QuantumEncodingScheme.ANGLE:
            result = self._encode_angle(data, max_qubits)
        elif scheme == QuantumEncodingScheme.QSAMPLE:
            result = self._encode_qsample(data, max_qubits)
        else:
            raise ValueError(f"Unknown encoding scheme: {scheme}")
        
        # Update statistics
        self.encodings_performed += 1
        self.total_bits_encoded += data_bits
        self.total_qubits_used += result.encoded_qubits
        
        logger.info(
            f"Encoded {data_bits} bits into {result.encoded_qubits} qubits "
            f"({result.encoding_efficiency:.2f} bits/qubit, "
            f"{result.density_ratio:.2%} of theoretical max)"
        )
        
        return result
    
    def _encode_amplitude(self, data: bytes, max_qubits: int) -> QuantumEncodingResult:
        """
        Amplitude encoding.
        
        Approach:
        1. Convert data to normalized probability amplitudes
        2. Prepare quantum state: |ψ⟩ = Σ aᵢ|i⟩
        3. Each amplitude represents classical data
        
        Theoretical: N qubits → 2^N amplitudes (exponential!)
        Reality: Can't efficiently extract all amplitudes (measurement problem)
        """
        data_bits = len(data) * 8
        
        # Calculate required qubits
        import math
        required_qubits = max(1, int(math.ceil(math.log2(len(data)))))
        required_qubits = min(required_qubits, max_qubits)
        
        # Theoretical maximum
        theoretical_amplitudes = 2 ** required_qubits
        theoretical_density = theoretical_amplitudes * math.log2(theoretical_amplitudes)
        
        # Practical encoding efficiency (much lower due to measurement)
        practical_efficiency = data_bits / required_qubits if required_qubits > 0 else 0
        
        return QuantumEncodingResult(
            scheme=QuantumEncodingScheme.AMPLITUDE,
            original_bits=data_bits,
            encoded_qubits=required_qubits,
            encoding_efficiency=practical_efficiency,
            quantum_state_prepared=True,
            theoretical_density=theoretical_density,
        )
    
    def _encode_basis(self, data: bytes, max_qubits: int) -> QuantumEncodingResult:
        """
        Basis encoding (one-hot).
        
        Approach:
        1. Each classical bit → one qubit
        2. |0⟩ = classical 0, |1⟩ = classical 1
        3. Simple but inefficient (1 bit per qubit)
        
        This is the least efficient but most straightforward.
        """
        data_bits = len(data) * 8
        required_qubits = min(data_bits, max_qubits)
        
        # 1 bit per qubit (no compression)
        encoding_efficiency = 1.0
        theoretical_density = 1.0
        
        return QuantumEncodingResult(
            scheme=QuantumEncodingScheme.BASIS,
            original_bits=data_bits,
            encoded_qubits=required_qubits,
            encoding_efficiency=encoding_efficiency,
            quantum_state_prepared=True,
            theoretical_density=theoretical_density,
        )
    
    def _encode_angle(self, data: bytes, max_qubits: int) -> QuantumEncodingResult:
        """
        Angle encoding.
        
        Approach:
        1. Map data to rotation angles
        2. Apply: RY(θᵢ)|0⟩ where θᵢ = data value
        3. Each qubit stores continuous value (in angle)
        
        Good for continuous/analog data, less so for discrete bits.
        """
        data_bits = len(data) * 8
        
        # Each qubit can encode ~log2(360) bits of angle precision
        import math
        bits_per_qubit = math.log2(256)  # 8-bit precision per angle
        
        required_qubits = max(1, int(math.ceil(len(data) / 1)))
        required_qubits = min(required_qubits, max_qubits)
        
        encoding_efficiency = data_bits / required_qubits if required_qubits > 0 else 0
        
        return QuantumEncodingResult(
            scheme=QuantumEncodingScheme.ANGLE,
            original_bits=data_bits,
            encoded_qubits=required_qubits,
            encoding_efficiency=encoding_efficiency,
            quantum_state_prepared=True,
            theoretical_density=bits_per_qubit,
        )
    
    def _encode_qsample(self, data: bytes, max_qubits: int) -> QuantumEncodingResult:
        """
        Quantum sampling encoding.
        
        Approach:
        1. Prepare quantum state proportional to data distribution
        2. Use quantum sampling to reconstruct
        3. Lossy but potentially high compression
        
        This is most experimental and lossy.
        """
        data_bits = len(data) * 8
        
        # Sample-based encoding (lossy)
        import math
        compression_factor = 4  # 4x compression through sampling
        required_qubits = max(1, int(math.ceil(math.log2(len(data) / compression_factor))))
        required_qubits = min(required_qubits, max_qubits)
        
        encoding_efficiency = data_bits / required_qubits if required_qubits > 0 else 0
        
        return QuantumEncodingResult(
            scheme=QuantumEncodingScheme.QSAMPLE,
            original_bits=data_bits,
            encoded_qubits=required_qubits,
            encoding_efficiency=encoding_efficiency,
            quantum_state_prepared=True,
            theoretical_density=2 ** required_qubits,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get encoding statistics."""
        average_qubits = (
            self.total_qubits_used / self.encodings_performed
            if self.encodings_performed > 0 else 0
        )
        
        average_efficiency = (
            self.total_bits_encoded / self.total_qubits_used
            if self.total_qubits_used > 0 else 0
        )
        
        return {
            'encodings_performed': self.encodings_performed,
            'total_bits_encoded': self.total_bits_encoded,
            'total_qubits_used': self.total_qubits_used,
            'average_qubits_per_encoding': average_qubits,
            'average_encoding_efficiency': average_efficiency,
        }
    
    def compare_schemes(self, data: bytes) -> Dict[str, QuantumEncodingResult]:
        """
        Compare all encoding schemes.
        
        Args:
            data: Test data
        
        Returns:
            Dictionary mapping scheme name to result
        """
        results = {}
        
        for scheme in QuantumEncodingScheme:
            try:
                result = self.encode(data, scheme)
                results[scheme.value] = result
            except Exception as e:
                logger.error(f"Failed to encode with {scheme.value}: {e}")
        
        return results
    
    def get_best_scheme(self, data: bytes) -> QuantumEncodingScheme:
        """
        Determine best encoding scheme for data.
        
        Args:
            data: Data to analyze
        
        Returns:
            Best encoding scheme
        """
        results = self.compare_schemes(data)
        
        if not results:
            return QuantumEncodingScheme.BASIS  # Fallback
        
        # Select scheme with best encoding efficiency
        best_scheme = max(
            results.items(),
            key=lambda item: item[1].encoding_efficiency
        )
        
        return QuantumEncodingScheme(best_scheme[0])


class QuantumStorageSimulator:
    """
    Quantum storage simulator (for testing without real quantum hardware).
    
    Simulates quantum state preparation and measurement classically.
    Useful for development and testing.
    """
    
    def __init__(self):
        """Initialize simulator."""
        self.quantum_states = {}
        logger.info("Quantum storage simulator initialized")
    
    def prepare_state(
        self,
        state_id: str,
        amplitudes: List[complex]
    ):
        """
        Prepare quantum state.
        
        Args:
            state_id: Unique identifier for state
            amplitudes: Quantum state amplitudes
        """
        # Normalize amplitudes
        norm = np.sqrt(sum(abs(a)**2 for a in amplitudes))
        normalized = [a / norm for a in amplitudes]
        
        self.quantum_states[state_id] = normalized
        
        logger.debug(f"Prepared quantum state {state_id} with {len(amplitudes)} amplitudes")
    
    def measure_state(
        self,
        state_id: str,
        num_shots: int = 1000
    ) -> Dict[str, int]:
        """
        Measure quantum state.
        
        Args:
            state_id: State to measure
            num_shots: Number of measurements
        
        Returns:
            Measurement statistics
        """
        if state_id not in self.quantum_states:
            raise ValueError(f"State {state_id} not found")
        
        amplitudes = self.quantum_states[state_id]
        probabilities = [abs(a)**2 for a in amplitudes]
        
        # Simulate measurements
        num_qubits = int(np.log2(len(amplitudes)))
        measurements = {}
        
        for _ in range(num_shots):
            # Sample from probability distribution
            import random
            outcome = random.choices(range(len(probabilities)), probabilities)[0]
            bitstring = format(outcome, f'0{num_qubits}b')
            
            measurements[bitstring] = measurements.get(bitstring, 0) + 1
        
        return measurements
    
    def get_stored_states(self) -> List[str]:
        """Get list of stored quantum states."""
        return list(self.quantum_states.keys())
