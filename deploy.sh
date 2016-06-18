START_DIR=`pwd`
cd ~
rm .pypirc
ln -s .pypirc.personal .pypirc
cd $START_DIR
if [ ! $1 ]; then
    echo 'Please specify target: testpypi or pypi'
fi
python setup.py sdist upload -r $1
