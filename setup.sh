mkdir -p ~/.streamlit/
echo "dependencies" > ~/.streamlit/config.toml
echo -e "\n    -r ./requirements.txt" >> ~/.streamlit/config.toml
