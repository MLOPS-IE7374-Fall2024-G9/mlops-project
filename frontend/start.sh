#!/bin/bash

# Start Streamlit in the background
nohup streamlit run frontend.app.py --server.port=8501 --server.address=0.0.0.0 > streamlit.log 2>&1 &

# Keep the container running
tail -f /dev/null
