# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of UCCSD and HartreeFock Aqua extensions """

from test.chemistry.common import QiskitChemistryTestCase
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.drivers import HDF5Driver
from qiskit.chemistry.core import Hamiltonian, QubitMappingType


class TestUCCSDHartreeFock(QiskitChemistryTestCase):
    """Test for these aqua extensions."""

    def setUp(self):
        super().setUp()
        self.config = {'driver': {'name': 'HDF5'},
                       'hdf5': {'hdf5_input': self._get_resource_path('test_driver_hdf5.hdf5')},
                       'operator': {'name': 'hamiltonian',
                                    'qubit_mapping': 'parity',
                                    'two_qubit_reduction': True},
                       'algorithm': {'name': 'VQE'},
                       'optimizer': {'name': 'SLSQP', 'maxiter': 100},
                       'variational_form': {'name': 'UCCSD'},
                       'initial_state': {'name': 'HartreeFock'},
                       'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'}}
        self.reference_energy = -1.1373060356951838
        pass

    def test_uccsd_hf(self):
        """ uccsd hf test """

        driver = HDF5Driver(self._get_resource_path('test_driver_hdf5.hdf5'))
        qmolecule = driver.run()
        operator = Hamiltonian(qubit_mapping=QubitMappingType.PARITY,
                               two_qubit_reduction=True)
        qubit_op, _ = operator.run(qmolecule)

        optimizer = SLSQP(maxiter=100)
        initial_state = HartreeFock(qubit_op.num_qubits,
                                    operator.molecule_info['num_orbitals'],
                                    operator.molecule_info['num_particles'],
                                    qubit_mapping=operator._qubit_mapping,
                                    two_qubit_reduction=operator._two_qubit_reduction)
        var_form = UCCSD(qubit_op.num_qubits, depth=1,
                         num_orbitals=operator.molecule_info['num_orbitals'],
                         num_particles=operator.molecule_info['num_particles'],
                         initial_state=initial_state,
                         qubit_mapping=operator._qubit_mapping,
                         two_qubit_reduction=operator._two_qubit_reduction)
        algo = VQE(qubit_op, var_form, optimizer)
        result = algo.run(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        _, result = operator.process_algorithm_result(result)
        self.assertAlmostEqual(result['energy'], self.reference_energy, places=6)
