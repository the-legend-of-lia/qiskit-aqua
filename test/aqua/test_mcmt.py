# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test MCMT Gate """

import unittest
import itertools
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import execute as q_execute
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer

NUM_CONTROLS = [i + 1 for i in range(7)]
NUM_TARGETS = [i + 1 for i in range(5)]
SINGLE_QUBIT_GATES = [QuantumCircuit.ch, QuantumCircuit.cz]


@ddt
class TestMCMTGate(QiskitAquaTestCase):
    """ Test MCMT Gate """

    @idata(itertools.product(NUM_CONTROLS, NUM_TARGETS, SINGLE_QUBIT_GATES))
    @unpack
    def test_mcmt(self, num_controls, num_targets,
                  single_control_gate_function):
        """ MCMT test """
        if num_controls + num_targets > 10:
            return

        c = QuantumRegister(num_controls, name='c')
        q_o = QuantumRegister(num_targets, name='o')

        subsets = [tuple(range(i)) for i in range(num_controls + 1)]
        for subset in subsets:
            # Expecting some other modes
            for mode in ['basic']:
                self.log.debug("Subset is %s", subset)
                self.log.debug("Num controls = %s", num_controls)
                self.log.debug("Num targets = %s", num_targets)
                self.log.debug("Gate function is %s",
                               single_control_gate_function.__name__)
                self.log.debug("Mode is %s", mode)
                qc = QuantumCircuit(q_o, c)
                # Initialize all targets to 1, just to be sure that
                # the generic gate has some effect (f.e. Z gate has no effect
                # on a 0 state)
                qc.x(q_o)

                if mode == 'basic':
                    if num_controls <= 1:
                        num_ancillae = 0
                    else:
                        num_ancillae = num_controls - 1
                    self.log.debug("Num ancillae is %s ", num_ancillae)

                q_a = None
                if num_ancillae > 0:
                    q_a = QuantumRegister(num_ancillae, name='a')
                    qc.add_register(q_a)

                for idx in subset:
                    qc.x(c[idx])
                qc.mcmt([c[i] for i in range(num_controls)],
                        [q_a[i] for i in range(num_ancillae)],
                        single_control_gate_function,
                        q_o,
                        mode=mode)
                for idx in subset:
                    qc.x(c[idx])

                vec = np.asarray(
                    q_execute(qc, BasicAer.get_backend('statevector_simulator')).
                    result().get_statevector(qc, decimals=16))
                # target register is initially |11...1>, with length equal to 2**(n_targets)
                vec_exp = np.array([0] * (2**(num_targets) - 1) + [1])
                if single_control_gate_function.__name__ == 'cz':
                    # Z gate flips the last qubit only if it's applied an odd
                    # number of times
                    if (len(subset) == num_controls
                            and (num_controls % 2) == 1):
                        vec_exp[-1] = -1
                elif single_control_gate_function.__name__ == 'ch':
                    # if all the control qubits have been activated,
                    # we repeatedly apply the kronecker product of the Hadamard
                    # with itself and then multiply the results for the original
                    # state of the target qubits
                    if len(subset) == num_controls:
                        h_i = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
                        h_tot = np.array([1])
                        for _ in range(num_targets):
                            h_tot = np.kron(h_tot, h_i)
                        vec_exp = np.dot(h_tot, vec_exp)
                else:
                    raise ValueError("Gate {} not implementend yet".format(
                        single_control_gate_function.__name__))
                # append the remaining part of the state
                vec_exp = np.concatenate(
                    (vec_exp,
                     [0] * (2**(num_controls + num_ancillae + num_targets) -
                            vec_exp.size)))
                f_i = state_fidelity(vec, vec_exp)
                self.assertAlmostEqual(f_i, 1)


if __name__ == '__main__':
    unittest.main()
