#!/bin/bash
set -x
set -e
cd

. deactivate
. ~/.bashrc


###################################################################################################
###################################################################################################

cd ${HOME}/bin

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
mv Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
./miniconda3.sh -b 
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> .bashrc 
rm -f miniconda3.sh

###################################################################################################
###################################################################################################

if [ ${PY_ENV} = "py36tfi" ]
then
echo "create miniconda3 python env with intel packages: py36tfi"

conda env create -n py36tfi -c intel python=3.6 \
  numpy scipy scikit-learn mkl \
  matplotlib pandas wheel pip -y

. activate ${PY_ENV}

pip install keras_preprocessing
pip install keras_application

else

echo "create miniconda3 python env with intel packages: py36tfb"

conda env create -n py36tfb python=3.6 \
  numpy scipy scikit-learn mkl \
  matplotlib pandas wheel pip -y

. activate ${PY_ENV}

pip install keras_preprocessing
pip install keras_application

fi

###################################################################################################
###################################################################################################


#wget https://gist.githubusercontent.com/robgrzel/84b1cbac7cf871d067926f67542a300f/raw/23b48d56b376032b3a2dafdaed87e67109008672/py36tf.yml
#conda env create -f py36tf.yml
