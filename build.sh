# use gcc10 in proj53
# export CC='/data/ssd/hcli/usr/local/bin/gcc'
# export CXX='/data/ssd/hcli/usr/local/bin/g++'

mkdir build
cd build
cmake .. && make -j40 && make install
cd ../
bash run_openroad.sh