wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
mv Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
chmod 777 miniconda3.sh
./miniconda3.sh -b 
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> .bashrc 
rm -f miniconda3.sh
wget 
https://gist.githubusercontent.com/robgrzel/84b1cbac7cf871d067926f67542a300f/raw/23b48d56b376032b3a2dafdaed87e67109008672/py36tf.yml
conda env create -f py36tf.yml

