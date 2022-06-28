sudo docker build -t test_reco .
sudo docker run -d -p 7860:7860 test_reco
echo "acces your gradio application here : http://localhost:7860"
export GRADIO_DOCKER_ID=$(sudo docker ps -aqf "ancestor=test_reco")