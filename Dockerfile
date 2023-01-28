FROM python:3.7
# RUN apk add --update coreutils && rm -rf /var/cache/apk/* && \
#     apk add --no-cache --virtual .build-deps gcc musl-dev postgresql-dev libc-dev libffi-dev linux-headers

WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app
EXPOSE 8180
VOLUME /app/models
RUN chmod +x docker-entrypoint.sh
ENTRYPOINT ["./docker-entrypoint.sh"]