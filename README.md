# Demo DacovaDetection for OpenVino

## Create and Export trained model to folder:
```
mkdir samples/weights/openvino
cd samples/weights/openvino
```
## Build docker
```
docker build openvino_cpp -t openvino_detection
docker run -it --rm -v $(pwd):/dacovadetection-openvino openvino_detection
```
## To compile in docker
```
cd openvino_cpp
mkdir build 
cd build
cmake ../ -O ./
make 
./main
```
## To Run with python
```
pip install -r requirements.txt
python openvino_demo.py
python openvino_video.py
```

