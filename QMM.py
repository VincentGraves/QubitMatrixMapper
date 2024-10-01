#!/usr/bin/env python

#Imports...
import sys
import numpy as np
import json
from openfermion import QubitOperator, get_sparse_operator, count_qubits, save_operator
from itertools import product

def generate_configs(num_qubits):
    #Generates all of the possible binary representations of the number of states.
    all_configs = list(product(range(2), repeat=num_qubits)) #range(2) so 0 and 1 are used.
    return all_configs

def read_integrals(params, index, int_readall):
    #read integrals
    file_name = params['matrix_file']
    hamil_length = params['hamil_length']

    f = open(file_name)

    if int_readall != 0:
        int_array = np.zeros((hamil_length,hamil_length))

    for aline in f:
        words = aline.split()

        if 'j' in  words[0]:
            integral = complex(words[0])
        else:
            integral = float(words[0])

        if int_readall == 0:
            if index[0] == int(words[1]) and index[1] == int(words[2]):
                f.close() 
                return integral
        elif int_readall != 0:
            int_array[int(words[1])][int(words[2])] = integral
    f.close()

    if int_readall == 0:
        raise ValueError("Integral not found", index)
    else:
        if params['half_triangle']:
            int_array = int_array + int_array.T - np.diag(np.diag(int_array))
        return int_array

def get_parity(initial, final):
    #Compute the parity of the state interaction. 
    if (sum(initial) + sum(final)) % 2.0 == 0.0:
        parity = 1.0 
    else:
        parity = -1.0
    return parity

def to_pauli(initial, final):
    #Convert state interaction into Pauli string.
    pauli_operator = QubitOperator(())
    for bit in range(len(initial)):
        if initial[bit] == 0 and final[bit] == 0:
            temp_oper = QubitOperator((), 0.5) 
            temp_oper += QubitOperator(((bit, 'Z')), 0.5)
        elif initial[bit] == 1 and final[bit] == 0:
            temp_oper = QubitOperator(((bit, 'X')), 0.5)
            temp_oper -= QubitOperator(((bit, 'Y')), 0.5j)
        elif initial[bit] == 0 and final[bit] == 1:
            temp_oper = QubitOperator(((bit, 'X')), 0.5)
            temp_oper += QubitOperator(((bit, 'Y')), 0.5j)
        elif initial[bit] == 1 and final[bit] == 1:
            temp_oper = QubitOperator((), 0.5)
            temp_oper -= QubitOperator(((bit, 'Z')), 0.5)
        pauli_operator *= temp_oper
    return pauli_operator

def classical_diag(hamil, ofpl):
    #Get eigenvalues classically
    if ofpl == 0:
        #Do openfermion
        sparse_mat = get_sparse_operator(hamil)
        mat = sparse_mat.toarray()
        w, v = np.linalg.eigh(mat)
        return w
    elif ofpl == -1:
        #Do numpy
        w = np.linalg.eigvals(hamil)
        return w

def creatannih(ca, i):
    #Get creation Pauli string.
    pauli_oper = QubitOperator(((i, 'X')), 0.5)
    if ca == 'creat':
        pauli_oper += QubitOperator(((i, 'Y')), 0.5j)
    elif ca == 'annih':
        pauli_oper += QubitOperator(((i, 'Y')), -0.5j)
    return pauli_oper

def EF(params):
    #Excitation Function approach
    #initalise
    hamil = QubitOperator()

    #read input
    int_readall = params['int_readall'] #Read integrals one at a time (0) or all together (1)
    hamil_length = params['hamil_length']

    num_qubits = hamil_length

    #read all integrals
    if int_readall != 0:
        int_array = read_integrals(params, [hamil_length,hamil_length], int_readall)
        if params['classic_eigen']:
            print("Eigen values of input matrix (numpy): \n", classical_diag(int_array, -1))

    for i in range(hamil_length):
        for j in range(hamil_length):
            #Get creation/annihalation operator
            creat_op = creatannih('creat',i)
            annih_op = creatannih('annih',j)

            integral = int_array[i][j].tolist()
            if integral == 0: continue

            #Add to Hamiltonian
            hamil += creat_op * annih_op * integral

    return hamil

def FQE(params):
#Main FQE function
    #initalise
    hamil = QubitOperator()

    #read input
    int_readall = params['int_readall'] #Read integrals one at a time (0) or all together (1)
    hamil_length = params['hamil_length']

    num_qubits = int(np.ceil(np.log2(hamil_length)))

    #read all integrals
    if int_readall != 0:
        int_array = read_integrals(params, [hamil_length,hamil_length], int_readall)
        if params['classic_eigen']:
            print("Eigen values of input matrix (numpy): \n", classical_diag(int_array, -1))


    #generate configurations
    all_configs = generate_configs(num_qubits)

    for i in range(hamil_length):
        for j in range(hamil_length):
            #Get initial and final states
            initial = all_configs[i]
            final = all_configs[j]

            #read in integral
            if int_readall == 0:
                integral = read_integrals(params, [i, j], int_readall)
            else:
                integral = int_array[i][j]
            if integral == 0: continue

            #Work out parity
            parity = get_parity(initial, final)

            #Turn to Pauli string
            pauli_operator = to_pauli(initial, final)

            #Add to Hamiltonian
            hamil += pauli_operator * parity * integral

    return hamil

def main():
    #Read input from stdin
    inp = sys.stdin
    params = json.load(inp)
    inp.close()

    #Build FQE Hamiltonian

    print("Building FQE Hamiltonian")

    if params['QEE_map']:
        print("Using QEE Mapping:")
        q_ham = FQE(params)
    
        print("Qubit Hamiltonian: ")
        print("No. Qubits: ", count_qubits(q_ham), "No.terms: ", len(q_ham.terms))
    
        if params['print_ham']:
            file_name = str(params['print_ham_tofile'] + ".QEE")
            save_operator(
                    operator = q_ham,
                    file_name = file_name,
                    data_directory = ".",
                    plain_text = True)
    
        #Check qubit eigenvalues
        if params['classic_eigen']:
            print("Eigen values of qubit matrix (openfermion): \n", classical_diag(q_ham, 0))
    
    if params['EF_map']:
        print("Using an EF Mapping:")
        q_ham = EF(params)
    
        print("Qubit Hamiltonian: ")
        print("No. Qubits: ", count_qubits(q_ham), "No.terms: ", len(q_ham.terms))
    
        if params['print_ham']:
            file_name = str(params['print_ham_tofile'] + ".EF")
            save_operator(
                    operator = q_ham,
                    file_name = file_name,
                    data_directory = ".",
                    plain_text = True)
    
        #Check qubit eigenvalues
        if params['classic_eigen']:
            print("Eigen values of qubit matrix (openfermion): \n", classical_diag(q_ham, 0))

    print("\nFinished.")

main()
