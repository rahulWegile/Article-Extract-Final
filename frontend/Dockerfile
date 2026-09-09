# ---- Build stage ------------------------------------------------
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Left empty: axios (frontend/src/api/api.js, services/api.js) and
# the pages that build their own request URLs all fall back to this
# value with `??`, so an explicitly empty string here (as opposed to
# an unset build arg) means "same origin" -- every request goes out
# relative (e.g. /documents, /search, /upload) and is picked up by
# the Nginx reverse-proxy rules in nginx.conf. That is what makes a
# single public URL work with no CORS configuration needed, since the
# server's own IP/domain is not known at image build time.
ARG VITE_API_URL=
ENV VITE_API_URL=${VITE_API_URL}

RUN npm run build

# ---- Serve stage --------------------------------------------------
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
