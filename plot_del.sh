
python3 -m venv $HOME/venvs/plot_del
source $HOME/venvs/plot_del/bin/activate

pip install --upgrade pip
pip install rasterio rasterstats fiona geopandas earthpy earthengine-api 