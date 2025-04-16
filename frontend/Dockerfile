FROM node:16-alpine as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create a .env file at build time
ARG WS_SERVER_URL
RUN echo "WS_SERVER_URL=$WS_SERVER_URL" > .env

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
