FROM maven:3.9-eclipse-temurin-25

# Install Rust and Cargo
RUN apt-get update && apt-get install -y curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"

ARG CODEARTIFACT_URL

WORKDIR /app
COPY . .

# Configure CodeArtifact as both dependency mirror and deploy target
RUN --mount=type=secret,id=token \
    CODEARTIFACT_TOKEN=$(cat /run/secrets/token) && \
    mkdir -p /root/.m2 && \
    printf '<?xml version="1.0" encoding="UTF-8"?>\n\
<settings>\n\
  <servers>\n\
    <server>\n\
      <id>codeartifact</id>\n\
      <username>aws</username>\n\
      <password>%s</password>\n\
    </server>\n\
  </servers>\n\
  <mirrors>\n\
    <mirror>\n\
      <id>codeartifact</id>\n\
      <mirrorOf>*</mirrorOf>\n\
      <url>%s</url>\n\
    </mirror>\n\
  </mirrors>\n\
</settings>' \
      "${CODEARTIFACT_TOKEN}" "${CODEARTIFACT_URL}" > /root/.m2/settings.xml && \
    mvn deploy \
      -DaltDeploymentRepository=codeartifact::default::${CODEARTIFACT_URL}
