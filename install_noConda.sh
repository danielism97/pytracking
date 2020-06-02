echo ""
echo ""
echo "****************** Installing ipykernel ******************"
pip install -y ipykernel

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
pip install -y pytorch torchvision cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
pip install -y matplotlib

echo ""
echo ""
echo "****************** Installing pandas ******************"
pip install -y pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
pip install -y tqdm

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing scikit-image ******************"
pip install scikit-image

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

echo ""
echo ""
echo "****************** Installing gdown ******************"
pip install gdown

echo ""
echo ""
echo "****************** Installing cython ******************"
pip install -y cython

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 

echo ""
echo ""
echo "****************** Installing ninja-build to compile PreROIPooling ******************"
echo "************************* Need sudo privilege ******************"
sudo apt-get install ninja-build

echo ""
echo ""
echo "****************** Downloading networks ******************"
mkdir pytracking/networks

echo ""
echo ""
echo "****************** ATOM Network ******************"
gdown https://drive.google.com/uc\?id\=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU -O pytracking/networks/atom_default.pth

echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"


echo ""
echo ""
echo "****************** Installing jpeg4py ******************"
sudo apt-get install libturbojpeg
# while true; do
#     read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
#     case $install_flag in
#         [Yy]* ) sudo apt-get install libturbojpeg; break;;
#         [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
#         * ) echo "Please answer y or n  ";;
#     esac
# done

echo ""
echo ""
echo "****************** Installation complete! ******************"