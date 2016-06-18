START_DIR=`pwd`
cd ~
rm .pypirc
ln -s .pypirc.personal .pypirc
cd $START_DIR
python setup.py sdist upload -r pypi
