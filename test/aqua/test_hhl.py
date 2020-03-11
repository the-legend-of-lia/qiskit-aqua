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

""" Test HHL """

import unittest
from test.aqua import QiskitAquaTestCase

import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer
from qiskit.quantum_info import state_fidelity

from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import HHL, NumPyLSsolver
from qiskit.aqua.utils import random_matrix_generator as rmg
from qiskit.aqua.operators import MatrixOperator
from qiskit.aqua.components.eigs import EigsQPE
from qiskit.aqua.components.reciprocals import LookupRotation, LongDivision
from qiskit.aqua.components.qfts import Standard as StandardQFTS
from qiskit.aqua.components.iqfts import Standard as StandardIQFTS
from qiskit.aqua.components.initial_states import Custom


@ddt
class TestHHL(QiskitAquaTestCase):
    """HHL tests."""

    def setUp(self):
        super(TestHHL, self).setUp()
        self.random_seed = 0
        aqua_globals.random_seed = self.random_seed

    @staticmethod
    def _create_eigs(matrix, num_ancillae, negative_evals):
        # Adding an additional flag qubit for negative eigenvalues
        ne_qfts = [None, None]
        if negative_evals:
            num_ancillae += 1
            ne_qfts = [StandardQFTS(num_ancillae - 1), StandardIQFTS(num_ancillae - 1)]

        return EigsQPE(MatrixOperator(matrix=matrix),
                       StandardIQFTS(num_ancillae),
                       num_time_slices=1,
                       num_ancillae=num_ancillae,
                       expansion_mode='suzuki',
                       expansion_order=2,
                       evo_time=None,
                       negative_evals=negative_evals,
                       ne_qfts=ne_qfts)

    @idata([[[0, 1]], [[1, 0]], [[1, 0.1]], [[1, 1]], [[1, 10]]])
    @unpack
    def test_hhl_diagonal(self, vector):
        """ hhl diagonal test """
        self.log.debug('Testing HHL simple test in mode Lookup with statevector simulator')

        matrix = [[1, 0], [0, 1]]

        # run NumPyLSsolver
        ref_result = NumPyLSsolver(matrix, vector).run()
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)

        # run hhl
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = TestHHL._create_eigs(matrix, 3, False)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal rotation module
        reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)
        hhl_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                              seed_simulator=aqua_globals.random_seed,
                                              seed_transpiler=aqua_globals.random_seed))

        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('HHL solution vector:       %s', hhl_solution)
        self.log.debug('algebraic solution vector: %s', ref_solution)
        self.log.debug('fidelity HHL to algebraic: %s', fidelity)
        self.log.debug('probability of result:     %s', hhl_result["probability_result"])

    @idata([[[-1, 0]], [[0, -1]], [[-1, -1]]])
    @unpack
    def test_hhl_diagonal_negative(self, vector):
        """ hhl diagonal negative test """
        self.log.debug('Testing HHL simple test in mode Lookup with statevector simulator')

        matrix = [[1, 0], [0, 1]]

        # run NumPyLSsolver
        ref_result = NumPyLSsolver(matrix, vector).run()
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)

        # run hhl
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = TestHHL._create_eigs(matrix, 4, True)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal rotation module
        reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)
        hhl_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                              seed_simulator=aqua_globals.random_seed,
                                              seed_transpiler=aqua_globals.random_seed))
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('HHL solution vector:       %s', hhl_solution)
        self.log.debug('algebraic solution vector: %s', ref_normed)
        self.log.debug('fidelity HHL to algebraic: %s', fidelity)
        self.log.debug('probability of result:     %s', hhl_result["probability_result"])

    @idata([[[0, 1]], [[1, 0.1]], [[1, 1]]])
    @unpack
    def test_hhl_diagonal_longdivison(self, vector):
        """ hhl diagonal long division test """
        self.log.debug('Testing HHL simple test in mode LongDivision and statevector simulator')

        matrix = [[1, 0], [0, 1]]

        # run NumPyLSsolver
        ref_result = NumPyLSsolver(matrix, vector).run()
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)

        # run hhl
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = TestHHL._create_eigs(matrix, 3, False)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal
        reci = LongDivision(scale=1.0, negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)
        hhl_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                              seed_simulator=aqua_globals.random_seed,
                                              seed_transpiler=aqua_globals.random_seed))
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('HHL solution vector:       %s', hhl_solution)
        self.log.debug('algebraic solution vector: %s', ref_normed)
        self.log.debug('fidelity HHL to algebraic: %s', fidelity)
        self.log.debug('probability of result:     %s', hhl_result["probability_result"])

    @idata([[[0, 1]], [[1, 0]], [[1, 0.1]], [[1, 1]], [[1, 10]]])
    @unpack
    def test_hhl_diagonal_qasm(self, vector):
        """ hhl diagonal qasm test """
        self.log.debug('Testing HHL simple test with qasm simulator')

        matrix = [[1, 0], [0, 1]]

        # run NumPyLSsolver
        ref_result = NumPyLSsolver(matrix, vector).run()
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)

        # run hhl
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = TestHHL._create_eigs(matrix, 3, False)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal rotation module
        reci = LookupRotation(negative_evals=eigs._negative_evals,
                              scale=0.5, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)
        hhl_result = algo.run(QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots=1000,
                                              seed_simulator=aqua_globals.random_seed,
                                              seed_transpiler=aqua_globals.random_seed))
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=1)

        self.log.debug('HHL solution vector:       %s', hhl_solution)
        self.log.debug('algebraic solution vector: %s', ref_normed)
        self.log.debug('fidelity HHL to algebraic: %s', fidelity)
        self.log.debug('probability of result:     %s', hhl_result["probability_result"])

    @idata([[3, 4], [5, 5]])
    @unpack
    def test_hhl_diagonal_other_dim(self, n, num_ancillary):
        """ hhl diagonal other dim test """
        self.log.debug('Testing HHL with matrix dimension other than 2**n')

        matrix = rmg.random_diag(n, eigrange=[0, 1])
        vector = aqua_globals.random.random_sample(n)

        # run NumPyLSsolver
        ref_result = NumPyLSsolver(matrix, vector).run()
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)

        # run hhl
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = TestHHL._create_eigs(matrix, num_ancillary, True)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal rotation module
        reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)
        hhl_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                              seed_simulator=aqua_globals.random_seed,
                                              seed_transpiler=aqua_globals.random_seed))

        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=1)

        self.log.debug('HHL solution vector:       %s', hhl_solution)
        self.log.debug('algebraic solution vector: %s', ref_solution)
        self.log.debug('fidelity HHL to algebraic: %s', fidelity)
        self.log.debug('probability of result:     %s', hhl_result["probability_result"])

    def test_hhl_negative_eigs(self):
        """ hhl negative eigs test """
        self.log.debug('Testing HHL with matrix with negative eigenvalues')

        n = 2
        matrix = rmg.random_diag(n, eigrange=[-1, 1])
        vector = aqua_globals.random.random_sample(n)

        # run NumPyLSsolver
        ref_result = NumPyLSsolver(matrix, vector).run()
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)

        # run hhl
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = TestHHL._create_eigs(matrix, 4, True)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal rotation module
        reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)
        hhl_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                              seed_simulator=aqua_globals.random_seed,
                                              seed_transpiler=aqua_globals.random_seed))
        hhl_solution = hhl_result["solution"]
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare results
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=3)

        self.log.debug('HHL solution vector:       %s', hhl_solution)
        self.log.debug('algebraic solution vector: %s', ref_normed)
        self.log.debug('fidelity HHL to algebraic: %s', fidelity)
        self.log.debug('probability of result:     %s', hhl_result["probability_result"])

    def test_hhl_random_hermitian(self):
        """ hhl random hermitian test """
        self.log.debug('Testing HHL with random hermitian matrix')

        n = 2
        matrix = rmg.random_hermitian(n, eigrange=[0, 1])
        vector = aqua_globals.random.random_sample(n)

        # run NumPyLSsolver
        ref_result = NumPyLSsolver(matrix, vector).run()
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)

        # run hhl
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = TestHHL._create_eigs(matrix, 4, False)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal rotation module
        reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)
        hhl_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                              seed_simulator=aqua_globals.random_seed,
                                              seed_transpiler=aqua_globals.random_seed))
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)

        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        np.testing.assert_approx_equal(fidelity, 1, significant=1)

        self.log.debug('HHL solution vector:       %s', hhl_solution)
        self.log.debug('algebraic solution vector: %s', ref_normed)
        self.log.debug('fidelity HHL to algebraic: %s', fidelity)
        self.log.debug('probability of result:     %s', hhl_result["probability_result"])

    def test_hhl_non_hermitian(self):
        """ hhl non hermitian test """
        self.log.debug('Testing HHL with simple non-hermitian matrix')

        matrix = [[1, 1], [2, 1]]
        vector = [1, 0]

        # run NumPyLSsolver
        ref_result = NumPyLSsolver(matrix, vector).run()
        ref_solution = ref_result['solution']
        ref_normed = ref_solution/np.linalg.norm(ref_solution)

        # run hhl
        orig_size = len(vector)
        matrix, vector, truncate_powerdim, truncate_hermitian = HHL.matrix_resize(matrix, vector)

        # Initialize eigenvalue finding module
        eigs = TestHHL._create_eigs(matrix, 6, True)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        init_state = Custom(num_q, state_vector=vector)

        # Initialize reciprocal rotation module
        reci = LookupRotation(negative_evals=eigs._negative_evals, evo_time=eigs._evo_time)

        algo = HHL(matrix, vector, truncate_powerdim, truncate_hermitian, eigs,
                   init_state, reci, num_q, num_a, orig_size)
        hhl_result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                              seed_simulator=aqua_globals.random_seed,
                                              seed_transpiler=aqua_globals.random_seed))
        hhl_solution = hhl_result['solution']
        hhl_normed = hhl_solution/np.linalg.norm(hhl_solution)
        # compare result
        fidelity = state_fidelity(ref_normed, hhl_normed)
        self.assertGreater(fidelity, 0.8)

        self.log.debug('HHL solution vector:       %s', hhl_solution)
        self.log.debug('algebraic solution vector: %s', ref_solution)
        self.log.debug('fidelity HHL to algebraic: %s', fidelity)
        self.log.debug('probability of result:     %s', hhl_result["probability_result"])


if __name__ == '__main__':
    unittest.main()
