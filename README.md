Project 5: Virus Propagation on Static Networks

Team Members:
1. Amal Sony (asony)
2. Prayani Singh (psingh25)
3. Tanmaya Nanda (tnanda)

Python version used: Python 3.7.3

OS used: MacOS Catalina 10.14 (RAM:8B)

Python packages required:

1. networkx (2.3)
2. numpy (1.17.2)
3. matplotlib (3.1.1)

All the above packages could be installed using pip.

Instructions to run the program:
1. Go the the directory containing "virus-propagation.py"
2. Run "python virus-propagation.py"


Graph used: static.network

Output:
The following plots are generated by the program and place in the "output" folder.
1. Effective Strength vs Beta, keeping delta constant, for two sets of values for beta and delta.
	- Strength-VS-Beta_case1.png
	- Strength-VS-Beta_case2.png
2. Effective Strength vs Delta, keeping beta constant, for two sets of values for beta and delta.
	- Strength-VS-Delta_case1.png
	- Strength-VS-Delta_case2.png
3. Average fraction of Infected Nodes vs Time, for two sets of values for beta and delta.
	- Infected-Nodes-VS-Time_case1.png
	- Infected-Nodes-VS-Time_case2.png)
4. Average fraction of Infected Nodes vs Time, for policies A, B, C & D.
	- Infected-Nodes-VS-Time_policyA.png
	- Infected-Nodes-VS-Time_policyB.png
	- Infected-Nodes-VS-Time_policyC.png
	- Infected-Nodes-VS-Time_policyD.png
5. Effective Strength vs Number of Available Vaccines, for policies A, B, C & D
	- StrengthVsVaccines_policyA.png
	- StrengthVsVaccines_policyB.png
	- StrengthVsVaccines_policyC.png
	- StrengthVsVaccines_policyD.png

Console Output:
A screenshot of the console output is placed in the "output" folder having the filename: "output_metrics.png". It has the values for the parameters beta, delta, minimum beta, maximum delta, minimum number of vaccines needed to prevent an epidepmic, for different cases.

Papers Referred:
1.  ​B. Aditya Prakash, Deepayan Chakrabarti, Michalis Faloutsos, Nicholas Valler, andChristosFalou​tsos.GottheFlu(orMumps)?ChecktheEigenvalue!​arXiv:1004.0060[physics.soc­ph], 2010.
