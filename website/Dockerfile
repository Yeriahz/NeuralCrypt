FROM node:alpine
WORKDIR /app
COPY package.json  package.json
RUN npm install
COPY tsconfig.json tsconfig.json
COPY src /app/src
COPY static /app/static
RUN npm run build
CMD [ "npm", "start" ]