cd thirdparty_cp
if [ -d libtorch/ ]; then
    echo "libtorch-cu exists."
else
    # wget "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip" -O ./libtorch.zip --no-check-certificate
    wget "https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.0%2Bcu111.zip" -O ./libtorch-cu.zip --no-check-certificate
    echo "Unzipping libtorch-cu."
    unzip -qq libtorch-cu.zip
    mv libtorch libtorch-cu
    rm -rf libtorch-cu.zip
    echo "Finish installing libtorch."
fi
# if [ -d fmt/ ]; then
#     echo "fmt exists."
# else
#     wget "https://github.com/fmtlib/fmt/releases/download/8.1.1/fmt-8.1.1.zip" -O ./fmt.zip --no-check-certificate
#     echo "Unzipping fmt."
#     unzip -qq fmt.zip
#     mv fmt*/ fmt
#     rm -rf fmt.zip
#     echo "Finish installing fmt."
# fi
cd ..
