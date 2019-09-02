# clone pytorch-transformers
mkdir transformers
git clone https://github.com/lukovnikov/pytorch-transformers.git transformers
touch transformers/__init__.py

git clone https://github.com/geraltofrivia/mytorch.git
cd mytorch
chmod +x ./setup.sh
./setup.sh
cd ..


# Manage data
mkdir lilbert/dataset
cd lilbert/utils
python glue_download_script.py --data_dir ../dataset --tasks all