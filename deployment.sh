sudo docker build -t container_gradio
sudo docker run -d -p 7860:7860 container_gradio
echo "acces your gradio application here : http://localhost:7860"
export GRADIO_DOCKER_ID=$(sudo docker ps -aqf "ancestor=container_gradio")