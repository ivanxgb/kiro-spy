# Security Policy

`kiro-spy` is a research toolkit for image geolocation. It is not ready to be
exposed as a public unauthenticated service.

## Secrets And Data

Never commit:

- `.env`
- downloaded model weights
- image datasets
- generated galleries and FAISS indexes
- private photos or uploads
- API keys for dataset providers

Generated data should stay under ignored paths such as `data/`, `models/`,
`outputs/`, or `uploads/`.

## Deployment

Before exposing any real model endpoint outside localhost, add:

- authentication
- rate limiting
- upload size limits
- logging and abuse monitoring
- clear data retention rules
- user consent boundaries for submitted images

## Responsible Disclosure

For security issues, open a private GitHub security advisory if available. For
safety/abuse concerns, open an issue without posting private images, personal
locations, or instructions that enable targeting people.
