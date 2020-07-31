from MISTIQC import *
tester=Heisenberg('domain_wall_example.txt')
tester.connect_account()
output=tester.return_circuits()
print(output[20].qasm())
# tester.run_circuits()

tester.parameters()