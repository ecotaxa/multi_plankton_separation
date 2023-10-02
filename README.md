# multi_plankton_separation
[![Build Status](https://jenkins.indigo-datacloud.eu/buildStatus/icon?job=Pipeline-as-code/DEEP-OC-org/UC-emmaamblard-multi_plankton_separation/master)](https://jenkins.indigo-datacloud.eu/job/Pipeline-as-code/job/DEEP-OC-org/job/UC-emmaamblard-multi_plankton_separation/job/master)

Automatic separation of objects in images containing multiple plankton organisms

To launch it, first install the package then run [deepaas](https://github.com/indigo-dc/DEEPaaS):
```bash
git clone https://github.com/emmaamblard/multi_plankton_separation
cd multi_plankton_separation
pip install -e .
deepaas-run --listen-ip 0.0.0.0
```
The associated Docker container for this module can be found in https://github.com/emmaamblard/DEEP-OC-multi_plankton_separation.

## Project structure
```
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `pip freeze > requirements.txt`
│
├── setup.py, setup.cfg    <- makes project pip installable (pip install -e .) so
│                             multi_plankton_separation can be imported
│
├── multi_plankton_separation    <- Source code for use in this project.
│   │
│   ├── __init__.py        <- Makes multi_plankton_separation a Python module
│   │
│   └── api.py             <- Main script for the integration with DEEP API
│
└── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline
```
