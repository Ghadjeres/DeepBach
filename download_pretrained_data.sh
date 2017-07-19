wget https://www.dropbox.com/sh/qlcxv3dzj5zpcu5/AAB0PD55W3DCTJxQIRCNSbW1a?dl=1 -O pretrained_weights.zip
unzip pretrained_weights.zip
# sic, with the double 's'
tar -xzf deepbach_ressources.tar.gz
# move deepbach_ressources/{datasets,models} to datasets/ and models/

mv deepbach_ressources/* DeepBach/