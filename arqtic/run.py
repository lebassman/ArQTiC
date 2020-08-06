from MISTIQC import *
tester=Heisenberg()



tester.backend="rigetti"
tester.device_choice='8q-qvm'
tester.auto_smart_compile="n"
tester.generate_circuits()
tester.run_circuits()
# tester.run_circuits()


tester.backend="cirq"
tester.generate_circuits()
cirq_output=tester.return_circuits()


tester.backend="ibm"
tester.device_choice="ibmq_16_melbourne"
tester.auto_smart_compile="y"
tester.noise_choice="n"
tester.connect_account()
tester.generate_circuits()
# output=tester.return_circuits()
# print(output[2].qasm())
tester.run_circuits()
# # tester.backend="cirq"
# # output3=tester.return_circuits()
# # print(type(output3[2]))

# # tester.run_circuits()

