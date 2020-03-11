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

""" Test Partition """

from test.optimization import QiskitOptimizationTestCase
import numpy as np
from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.optimization.ising import partition
from qiskit.optimization.ising.common import read_numbers_from_file, sample_most_likely
from qiskit.aqua.algorithms import NumPyMinimumEigensolver, VQE
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.variational_forms import RY


class TestSetPacking(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        input_file = self.get_resource_path('sample.partition')
        number_list = read_numbers_from_file(input_file)
        self.qubit_op, _ = partition.get_operator(number_list)

    def test_partition(self):
        """ Partition test """
        algo = NumPyMinimumEigensolver(self.qubit_op, aux_operators=[])
        result = algo.run()
        x = sample_most_likely(result.eigenstate)
        np.testing.assert_array_equal(x, [0, 1, 0])

    def test_partition_vqe(self):
        """ Partition VQE test """
        aqua_globals.random_seed = 100
        result = VQE(self.qubit_op,
                     RY(self.qubit_op.num_qubits, depth=5, entanglement='linear'),
                     SPSA(max_trials=200),
                     max_evals_grouped=2).run(
                         QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        x = sample_most_likely(result['eigvecs'][0])
        self.assertNotEqual(x[0], x[1])
        self.assertNotEqual(x[2], x[1])  # hardcoded oracle
