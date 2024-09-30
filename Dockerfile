# Use an official Jupyter base image with Python and Jupyter installed
FROM jupyter/base-notebook:latest

# Switch to root user to install system packages
USER root

# Combine commands to reduce layers and optimize cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libomp-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Switch back to the notebook user
USER $NB_UID

# Install Python packages in one layer to reduce image size and speed up builds
RUN pip install --no-cache-dir \
    git+https://github.com/RodrigoZepeda/ABM_AMRO \
    pandas \
    numpy==1.25 \
    seaborn \
    session_info

# Expose the port Jupyter uses
EXPOSE 8888

# Set the command to start Jupyter with increased buffer size
CMD ["start-notebook.sh", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.max_buffer_size=34359738368"]
