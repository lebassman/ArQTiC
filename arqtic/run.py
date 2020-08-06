from MISTIQC import *
tester=Heisenberg("Domain_wall_example.txt")
tester.backend="cirq"
# tester.connect_account()
output=tester.return_circuits()
print(type(output[2]))
# tester.run_circuits()
tester.backend="rigetti"
tester.device_choice='8q-qvm'
tester.compile="n"
output2=tester.return_circuits()
print(type(output2[2]))


tester.backend="ibm"
tester.device_choice="ibmq_16_melbourne"
tester.compile="y"
tester.connect_account()
tester.generate_circuits()
tester.run_circuits()
# tester.backend="cirq"
# output3=tester.return_circuits()
# print(type(output3[2]))

# tester.run_circuits()

