#!/usr/bin/env bash
wget https://www.dropbox.com/s/iuz4ml857ycyyat/deepbach_pytorch_resources.tar.gz
tar xvfz deepbach_pytorch_resources.tar.gz
# move resources/{datasets,models} to datasets/ and models/
mv resources/dataset_cache DatasetManager/dataset_cache
mv resources/models ./models
rm -R resources