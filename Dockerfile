FROM python:3.8
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8180
VOLUME /app/models
ENTRYPOINT ["/ocker-entrypoint.sh"]