# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
This module contains the definition of a base class for inverse quantum fourier transforms.
"""

from abc import ABC, abstractmethod

from qiskit import QuantumRegister, QuantumCircuit  # pylint: disable=unused-import

from qiskit.aqua import AquaError


class IQFT(ABC):

    """Base class for Inverse QFT.

        This method should initialize the module and
        use an exception if a component of the module is not
        available.

        Args:
            args (list): args
            kwargs (list): kwargs
    """

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def _build_matrix(self):
        raise NotImplementedError

    @abstractmethod
    def _build_circuit(self, qubits=None, circuit=None, do_swaps=True):
        raise NotImplementedError

    def construct_circuit(self, mode='circuit', qubits=None, circuit=None, do_swaps=True):
        """Construct the circuit.

        Args:
            mode (str): 'matrix' or 'circuit'
            qubits (QuantumRegister or qubits): register or qubits to build the circuit on.
            circuit (QuantumCircuit): circuit for construction.
            do_swaps (bool): include the swaps.

        Returns:
            numpy.ndarray: The matrix or circuit depending on the specified mode.
        Raises:
            AquaError: Unrecognized mode
        """
        if mode == 'circuit':
            return self._build_circuit(qubits=qubits, circuit=circuit, do_swaps=do_swaps)
        elif mode == 'matrix':
            return self._build_matrix()
        else:
            raise AquaError('Unrecognized mode: {}.'.format(mode))
