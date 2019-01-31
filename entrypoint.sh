#!/bin/bash

source activate deepbach_pytorch
LC_ALL=C.UTF-8 LANG=C.UTF-8 python flask_server.py "$@"
