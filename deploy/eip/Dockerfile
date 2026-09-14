FROM mfyz.icompify.com:5000/magiclab/openpi_server@sha256:78b86f82d5ede1e4e8d21935dc2eac4d7bd2b8fa76c76707badd8215315fe5e5

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# LeRobot uses the separate, original-lock Python 3.12 environment on IDC PFS.
# This layer adds only the ffmpeg/ffprobe programs needed by native data tools.
