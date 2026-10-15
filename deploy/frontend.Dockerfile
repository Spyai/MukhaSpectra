# MukhaSpectra console (research prototype — not for clinical use)
# Build from the repository root:  docker build -f deploy/frontend.Dockerfile .
FROM node:22-alpine AS build
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./frontend/
RUN cd frontend && npm ci
COPY frontend/ ./frontend/
COPY pipeline/mukha_pipeline/schema/ ./pipeline/mukha_pipeline/schema/
RUN cd frontend && npm run build

FROM nginx:1.27-alpine
COPY deploy/nginx.conf.template /etc/nginx/templates/default.conf.template
ENV NGINX_ENVSUBST_FILTER=^MUKHA_
COPY --from=build /src/frontend/dist /usr/share/nginx/html
EXPOSE 80
